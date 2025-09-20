import logging
import os
import re
import signal
import threading
from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from queue import Full, Queue
from typing import Any

import psutil
import Pyro5.server
from PIL import Image

from imgsearch import config as cfg
from imgsearch.storage import VectorDB
from imgsearch.utils import bold, bytes2img, get_logger, print_err, print_inf

Pyro5.config.COMPRESSION = True  # type: ignore
Image.MAX_IMAGE_PIXELS = 100_000_000
cfg.BASE_DIR.mkdir(parents=True, exist_ok=True)
NETLOC = re.compile(r'^(?P<host>[^:]+):(?P<port>\d+)$')


@Pyro5.server.expose
class RPCService:
    """
    RPC Service for image search operations

    This service provides a Pyro5-based RPC interface for image search operations,
    including adding images to the index, searching by image or text, and comparing images.
    All methods are thread-safe using a simple lock.
    """

    def __init__(
        self,
        base_dir: Path = cfg.BASE_DIR,
        model_key: str = cfg.DEFAULT_MODEL_KEY,
        logger=None,
    ):
        """
        Initialize the RPC service with async processing.

        Args:
            base_dir: Path to the database base directory
            model_key: Key of the CLIP model to use
            logger: Logger instance (optional, passed from Server)
        """
        self.base_dir = base_dir
        self.model_key = model_key
        self.databases: dict[str, VectorDB] = {}
        self._lock = threading.Lock()
        self.logger = logger or get_logger('ImgSearchService', logging.INFO)

        # Async processing queue - now includes db_name
        self.image_queue: Queue[tuple[str, Image.Image, str]] = Queue(maxsize=cfg.BATCH_SIZE * 3)
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

        # Search concurrency control
        self.max_concurrent_searches = max(2, cfg.BATCH_SIZE // 2)
        self.search_semaphore = threading.Semaphore(self.max_concurrent_searches)

    @cached_property
    def clip(self):
        """Get CLIP model instance"""
        from imgsearch.clip import Clip

        return Clip(model_key=self.model_key)

    def _get_db(self, db_name: str = cfg.DB_NAME):
        """Get database instance"""
        if db_name not in self.databases:
            self.logger.debug(f'Loading database: {db_name}')
            self.databases[db_name] = VectorDB(db_name, self.base_dir)
        return self.databases[db_name]

    def handle_check_exist_labels(self, labels: Sequence[str], db_name: str = cfg.DB_NAME) -> list[bool]:
        """Check if multiple labels exist in the database"""
        db = self._get_db(db_name)
        return db.has_labels(*labels)

    def _process_images(self, images: list[Image.Image], labels: list[str], db_name: str):
        """Process a batch of images asynchronously."""
        self.logger.debug(f'Processing batch of {len(images)} images ({db_name})')
        try:
            features = self.clip.embed_images(images)

            with self._lock:
                db = self._get_db(db_name)
                db.add_items(labels, features)
                db.save()

            self.logger.debug(f'Added {len(images)} images for db "{db_name}" ({db.size=})')

        except Exception as e:
            self.logger.error(f'Failed to process batch: {e} ({e.__class__.__name__})')

    def _process_queue(self) -> None:
        """Background thread to process images from queue."""
        # Group images by database name
        batches: dict[str, tuple[list[Image.Image], list[str]]] = defaultdict(lambda: ([], []))

        while True:
            try:
                # Get image from queue
                label, image, db_name = self.image_queue.get(timeout=1)

                batch_images, batch_labels = batches[db_name]
                batch_images.append(image)
                batch_labels.append(label)

                # Process batch when full
                if len(batch_images) >= cfg.BATCH_SIZE:
                    self._process_images(batch_images, batch_labels, db_name)
                    batches[db_name] = ([], [])

            except Exception:
                # Process remaining images in all batches
                for db_name, (batch_images, batch_labels) in batches.items():
                    if batch_images:
                        self._process_images(batch_images, batch_labels, db_name)
                        batches[db_name] = ([], [])

    def handle_add_images(self, images: dict[str, bytes], db_name: str = cfg.DB_NAME) -> int:
        """
        Add images to the search index.

        Args:
            images: Dictionary mapping labels to image bytes
            db_name: Name of the database to add images to

        Returns:
            Number of images queued for processing
        """
        self.logger.info(f'[AddImages] {len(images)} images received for db: {db_name}')

        queued_count = 0
        try:
            for label, image_bytes in images.items():
                image = bytes2img(image_bytes)
                self.image_queue.put((label, image, db_name))
                queued_count += 1
        except Full as e:
            self.logger.error(f'Queue full, dropping image: {e}')

        return queued_count

    def handle_search(
        self,
        query: Any,
        k: int = 10,
        similarity: int = 0,
        db_name: str = cfg.DB_NAME,
    ) -> list[tuple[str, float]] | None:
        """
        Search for similar images using image or text query.

        Args:
            query: Text description or image bytes to search for
            k: Number of results to return
            similarity: Minimum similarity threshold (0-100)
            db_name: Name of the database to search in

        Returns:
            List of tuples containing (image_path, similarity_percentage)
            Returns None if search queue is full
            Returns empty list if no results found
        """
        # Non-blocking concurrency control
        if not self.search_semaphore.acquire(blocking=False):
            self.logger.info(f'[Search] Rejected - {self.max_concurrent_searches} concurrent searches active')
            return None

        try:
            self.logger.info(f'[Search] type={type(query).__name__}, {k=}, {similarity=}, {db_name=}')

            # Handle Pyro5 serialization quirks
            if isinstance(query, str):
                # Text search
                feature = self.clip.embed_text(str(query))

            elif isinstance(query, bytes):
                # Image search
                img = bytes2img(query)  # convert bytes to PIL Image
                feature = self.clip.embed_image(img)

            elif isinstance(query, dict):
                # Handle dict type (likely from Pyro5 serialization)
                if 'data' in query and isinstance(query['data'], (bytes, str)):
                    # Handle base64 encoded image data
                    img_data = query['data']
                    if isinstance(img_data, str):
                        import base64

                        img_data = base64.b64decode(img_data)
                    img = bytes2img(img_data)
                    feature = self.clip.embed_image(img)
                else:
                    self.logger.error(f'Invalid dict format: {list(query.keys())}')
                    return []

            else:
                self.logger.error(f'Unsupported query type: {type(query)}, value: {repr(query)[:200]}')
                return []

            db = self._get_db(db_name)
            return db.search(feature, k, similarity)
        except Exception as e:
            self.logger.error(f'Text search failed: {e}')
            return []
        finally:
            self.search_semaphore.release()

    def handle_list_dbs(self) -> list[str]:
        """
        List all available database names.

        Returns:
            List of database names found in the base directory
        """
        try:
            db_list = self._get_db(cfg.DB_NAME).db_list()
            self.logger.debug(f'List dbs: {db_list}')
            return db_list
        except Exception as e:
            self.logger.error(f'Failed to list databases: {e}')
            return []

    def handle_get_db_info(self, db_name: str = cfg.DB_NAME) -> dict:
        """
        Get database information.

        Args:
            db_name: Name of the database to get info for

        Returns:
            Dictionary containing database statistics (base_dir, size, capacity)
        """
        self.logger.debug(f'Get db info: {db_name}')
        db = self._get_db(db_name)
        return {
            'base': str(db.base.resolve()),
            'size': db.size,
            'capacity': db.capacity,
        }

    def handle_clear_db(self, db_name: str = cfg.DB_NAME) -> bool:
        """
        Clear all images from the database.

        Args:
            db_name: Name of the database to clear

        Returns:
            True if successful, False otherwise
        """
        self.logger.warning(f'[Clear] Clear database: {db_name}')
        try:
            db = self._get_db(db_name)
            with self._lock:
                db.clear()
                self.logger.debug(f'Database {db_name} cleared')
                return True
        except Exception as e:
            self.logger.error(f'Failed to clear database: {e}')
            return False

    def handle_compare_images(self, ibytes1: bytes, ibytes2: bytes) -> float:
        """
        Compare similarity between two images.

        Args:
            ibytes1: Bytes of the first image
            ibytes2: Bytes of the second image

        Returns:
            Similarity percentage (0-100)
        """
        self.logger.info('[Compare] 2 images')

        try:
            img1, img2 = bytes2img(ibytes1), bytes2img(ibytes2)
            return self.clip.compare_images(img1, img2)
        except Exception as e:
            self.logger.error(f'Failed to compare images: {e}')
            return 0


def get_log_level(level: str) -> int:
    """Map log level string to logging level constant."""
    mapping = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }
    return mapping.get(level.lower(), logging.INFO)


class Server:
    """Server for the RPC service with pid file management and signal handling."""

    def __init__(
        self,
        base_dir: Path = cfg.BASE_DIR,
        model_key: str = cfg.DEFAULT_MODEL_KEY,
        bind: str = cfg.UNIX_SOCKET,
        log_level: str = 'info',
    ):
        self._shutdown_requested = False
        self._restart_requested = False
        self.base_dir = base_dir
        self.model_key = model_key
        self.bind = bind
        self.pid_file = self.base_dir / 'isearch.pid'

        # Setup logger
        self.logger = get_logger('ImgSearchService', level=get_log_level(log_level))

        # Create service
        self.service = RPCService(base_dir=self.base_dir, model_key=self.model_key, logger=self.logger)
        self.daemon: Pyro5.server.Daemon | None = None

    def create_daemon(self):
        """Create and configure Pyro5 Daemon instance."""
        if netloc := NETLOC.match(self.bind):
            # TCP connection
            host = netloc.group('host')
            port = int(netloc.group('port'))
            daemon = Pyro5.server.Daemon(host=host, port=port)
        else:
            # UDS connection
            uds_path = Path(self.bind)
            if uds_path.exists():
                uds_path.unlink()
            daemon = Pyro5.server.Daemon(unixsocket=self.bind)

        return daemon

    def _write_pid_file(self):
        """Write current process ID to pid file."""
        try:
            self.pid_file.write_text(str(os.getpid()))
        except Exception as e:
            self.logger.error(f'Failed to create PID file: {e}')
            raise

    def _read_pid_file(self) -> int | None:
        """Read pid from pid file."""
        if self.pid_file.exists():
            return int(self.pid_file.read_text().strip())
        return None

    def _remove_pid_file(self):
        """Remove pid file."""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.debug('PID file removed')
        except Exception as e:
            self.logger.error(f'Failed to remove PID file: {e}')

    @staticmethod
    def is_running(pid: int) -> bool:
        """Check if process with given pid is running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def run(self):
        """Run the server with proper signal handling."""
        # Configure Pyro5 to use msgpack serializer
        Pyro5.config.SERIALIZER = 'msgpack'  # type: ignore

        try:
            # Check if already running
            existing_pid = self._read_pid_file()
            if not existing_pid or not self.is_running(existing_pid):
                self.logger.info('Starting iSearch Service...')
            else:
                self.logger.error(f'Server already running with PID {existing_pid}')
                return

            # Register service and start daemon
            self.logger.info('Preloading CLIP model...')
            self.daemon = self.create_daemon()
            self.daemon.register(self.service, objectId=cfg.SERVICE_NAME)
            self.service.clip  # preload clip model  # noqa: B018

            # Create pid file
            self._write_pid_file()

            # Setup signal handlers
            signal.signal(signal.SIGTERM, self.handle_signal)
            signal.signal(signal.SIGHUP, self.handle_signal)
            signal.signal(signal.SIGINT, self.handle_signal)

            # Log service info
            self.logger.info('iSearch service started')
            self.logger.debug(f'Listening : {self.bind}')
            self.logger.debug(f'Serializer: {Pyro5.config.SERIALIZER}')
            self.logger.debug(f'Process ID: {os.getpid()}')
            self.logger.debug(f'Base dir  : {self.base_dir}')
            self.logger.debug(f'Model     : {self.model_key}')

            self.daemon.requestLoop()
        except ImportError as e:
            self.logger.error(f'Missing optional dependencies: {e}')
            self.logger.error("Run `pip install 'imgsearch[all]'` for a full install.")
        except Exception as e:
            self.logger.error(f'Server failed to start: {e}')
        finally:
            self.cleanup()
            self.logger.info('iSearch service stopped')

    def handle_signal(self, signum, _):
        """Handle various signals."""
        if self._shutdown_requested:
            return

        match signum:
            case signal.SIGTERM | signal.SIGINT:
                name = signal.Signals(signum).name
                self.logger.warning(f'Received {name}, shutting down now...')
                self._shutdown_requested = True
                if self.daemon:
                    self.daemon.shutdown()
            case signal.SIGHUP:
                self.logger.warning('Received SIGHUP, ignoring (restart not supported)')

    def stop(self):
        """Stop the running server."""
        pid = self._read_pid_file()
        if not pid:
            print_err('iSearch service is not running')
            return False

        if not self.is_running(pid):
            print_err(f'Process {pid} is not running')
            self._remove_pid_file()
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            print_err(f'Sent SIGTERM to process {pid}')
            return True
        except OSError as e:
            print_err(f'Failed to send SIGTERM: {e}')
            return False

    def cleanup(self):
        """Cleanup resources before exit."""
        # Save databases
        if isinstance(self.service, RPCService):
            for name, db in self.service.databases.items():
                try:
                    db.save()
                    self.logger.debug(f'Database "{name}" saved')
                except Exception as e:
                    self.logger.error(f'Failed to save db "{name}": {e}')

        # Shutdown daemon if running
        if self.daemon:
            try:
                self.daemon.shutdown()
                self.logger.debug('Daemon shutdown completed')
            except Exception as e:
                self.logger.error(f'Failed to shutdown daemon: {e}')

        # Cleanup socket
        if isinstance(self.bind, str):
            uds_path = Path(self.bind)
            if uds_path.is_socket():
                self.logger.debug(f'Cleaning up unix socket: {self.bind}')
                uds_path.unlink()

        # Remove pid file
        self._remove_pid_file()

    @classmethod
    def status(cls):
        """Check server status."""
        pid_file = cfg.BASE_DIR / 'isearch.pid'
        if not pid_file.exists():
            print_err('iSearch service is not running')
            return False

        try:
            pid = int(pid_file.read_text().strip())
            if cls.is_running(pid):
                print_inf('iSearch service is running', marked=True)
                print_inf(f'* {bold("PID")}: {pid}')
                try:
                    memory_info = psutil.Process(pid).memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    print_inf(f'* {bold("MEM")}: {memory_mb:.1f} MB')
                except Exception as e:
                    print_err(f'* {bold("MEM")}: unknown ({e})')
                return True
            else:
                print_err('PID file exists but process is not running')
                pid_file.unlink(missing_ok=True)
                return False
        except Exception as e:
            print_err(f'Error checking status: {e}')
            return False


def main():
    """Main function to run the RPC service directly."""
    server = Server()
    server.run()


if __name__ == '__main__':
    main()
