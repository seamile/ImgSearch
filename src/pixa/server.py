import logging
import os
import signal
import threading
from collections import defaultdict
from pathlib import Path
from queue import Full, Queue
from typing import Any

import Pyro5.server
from PIL import Image

from pixa.consts import BASE_DIR, BATCH_SIZE, DB_NAME, DEFAULT_MODEL, SERVICE_NAME, UNIX_SOCKET
from pixa.storage import VectorDB
from pixa.utils import bytes2img, get_logger, print_err

Image.MAX_IMAGE_PIXELS = 100_000_000
BASE_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger('PixaService', logging.INFO)


@Pyro5.server.expose
class RPCService:
    """
    RPC Service for image search operations

    This service provides a Pyro5-based RPC interface for image search operations,
    including adding images to the index, searching by image or text, and comparing images.
    All methods are thread-safe using a simple lock.
    """

    def __init__(self, base_dir: Path = BASE_DIR, model_name: str = DEFAULT_MODEL):
        """
        Initialize the RPC service with async processing.

        Args:
            base_dir: Path to the database base directory
            model_name: Name of the CLIP model to use
        """
        from pixa.clip import Clip  # import when needed

        self.clip = Clip(model_name)
        self.base_dir = base_dir
        self.databases: dict[str, VectorDB] = {}
        self._lock = threading.Lock()

        # Async processing queue - now includes db_name
        self.image_queue: Queue[tuple[str, Image.Image, str]] = Queue(maxsize=BATCH_SIZE * 5)
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

        # Search concurrency control
        self.max_concurrent_searches = max(2, BATCH_SIZE // 2)
        self.search_semaphore = threading.Semaphore(self.max_concurrent_searches)

    def _get_db(self, db_name: str = DB_NAME):
        """Get database instance"""
        if db_name not in self.databases:
            self.databases[db_name] = VectorDB(db_name, self.base_dir)
        return self.databases[db_name]

    def exists_in_db(self, label: str, db_name: str = DB_NAME):
        """Check if a label exists in the database"""
        db = self._get_db(db_name)
        return db.has_label(label)

    def _process_images(self, images: list[Image.Image], labels: list[str], db_name: str):
        """Process a batch of images asynchronously."""
        logger.debug(f'Processing batch of {len(images)} images ({db_name})')
        try:
            features = self.clip.embed_images(images)

            with self._lock:
                db = self._get_db(db_name)
                db.add_items(labels, features)
                db.save()

            logger.debug(f'Added {len(images)} images for db "{db_name}" ({db.size=})')

        except Exception as e:
            logger.error(f'Failed to process batch: {e} ({e.__class__.__name__})')

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
                if len(batch_images) >= BATCH_SIZE:
                    self._process_images(batch_images, batch_labels, db_name)
                    batches[db_name] = ([], [])

            except Exception:
                # Process remaining images in all batches
                for db_name, (batch_images, batch_labels) in batches.items():
                    if batch_images:
                        self._process_images(batch_images, batch_labels, db_name)
                        batches[db_name] = ([], [])

    def handle_add_images(self, images: dict[str, bytes], db_name: str = DB_NAME) -> int:
        """
        Add images to the search index.

        Args:
            images: Dictionary mapping labels to image bytes
            db_name: Name of the database to add images to

        Returns:
            Number of images queued for processing
        """
        logger.info(f'[AddImages] {len(images)} images received for db: {db_name}')

        queued_count = 0
        try:
            for label, image_bytes in images.items():
                image = bytes2img(image_bytes)
                self.image_queue.put((label, image, db_name))
                queued_count += 1
        except Full as e:
            logger.error(f'Queue full, dropping image: {e}')

        return queued_count

    def handle_search(
        self, query: Any, k: int = 10, similarity: float = 0.0, db_name: str = DB_NAME
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
            logger.info(f'[Search] Rejected - {self.max_concurrent_searches} concurrent searches active')
            return None

        try:
            logger.info(f'[Search] type={type(query).__name__}, {k=}, {similarity=}, {db_name=}')

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
                    logger.error(f'Invalid dict format: {list(query.keys())}')
                    return []

            else:
                logger.error(f'Unsupported query type: {type(query)}, value: {repr(query)[:200]}')
                return []

            db = self._get_db(db_name)
            return db.search(feature, k, similarity)
        except Exception as e:
            logger.error(f'Text search failed: {e}')
            return []
        finally:
            self.search_semaphore.release()

    def handle_get_db_info(self, db_name: str = DB_NAME) -> dict:
        """
        Get database information.

        Args:
            db_name: Name of the database to get info for

        Returns:
            Dictionary containing database statistics (base_dir, size, capacity)
        """
        db = self._get_db(db_name)
        return {
            'base': str(db.base.resolve()),
            'name': db_name,
            'size': db.size,
            'capacity': db.capacity,
        }

    def handle_clear_db(self, db_name: str = DB_NAME) -> bool:
        """
        Clear all images from the database.

        Args:
            db_name: Name of the database to clear

        Returns:
            True if successful, False otherwise
        """
        logger.warning(f'[Clear] Clear database: {db_name}')
        db = self._get_db(db_name)
        with self._lock:
            db.clear()
            logger.debug(f'Database {db_name} cleared')
            return True

    def handle_compare_images(self, path1: str, path2: str) -> float:
        """
        Compare similarity between two images.

        Args:
            path1: Path to the first image
            path2: Path to the second image

        Returns:
            Similarity percentage (0-100)
        """
        logger.info(f'[Compare] {path1} and {path2}')

        try:
            img1, img2 = Image.open(path1), Image.open(path2)
            return self.clip.compare_images(img1, img2)
        except Exception as e:
            logger.error(f'Failed to compare images: {e}')
            return 0


class Server:
    """Server for the RPC service with pid file management and signal handling."""

    def __init__(self, base_dir: Path = BASE_DIR, model_name: str = DEFAULT_MODEL):
        self.daemon = None
        self._shutdown_requested = False
        self._restart_requested = False
        self.service = None
        self.base_dir = base_dir
        self.model_name = model_name
        self.pid_file = self.base_dir / 'pixa.pid'

    def _write_pid_file(self):
        """Write current process ID to pid file."""
        try:
            self.pid_file.write_text(str(os.getpid()))
        except Exception as e:
            logger.error(f'Failed to create PID file: {e}')
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
                logger.debug('PID file removed')
        except Exception as e:
            logger.error(f'Failed to remove PID file: {e}')

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

        # Check if already running
        existing_pid = self._read_pid_file()
        if existing_pid and self.is_running(existing_pid):
            logger.error(f'Server already running with PID {existing_pid}')
            return

        if UNIX_SOCKET.exists():
            UNIX_SOCKET.unlink()

        try:
            logger.info('Starting Pixa Service...')

            # Create pid file
            self._write_pid_file()

            # Setup signal handlers
            signal.signal(signal.SIGTERM, self.handle_signal)
            signal.signal(signal.SIGHUP, self.handle_signal)
            signal.signal(signal.SIGINT, self.handle_signal)

            # Initialize service
            self.service = RPCService(base_dir=self.base_dir, model_name=self.model_name)
            self.daemon = Pyro5.server.Daemon(unixsocket=str(UNIX_SOCKET))
            self.daemon.register(self.service, objectId=SERVICE_NAME)

            # log service info
            logger.debug(f'Listening on: {UNIX_SOCKET}')
            logger.debug(f'Serializer: {Pyro5.config.SERIALIZER}')
            logger.debug(f'PID: {os.getpid()}')
            logger.debug(f'Base dir: {self.base_dir}')
            logger.debug(f'Model: {self.model_name}')
            logger.info('Pixa service started')

            self.daemon.requestLoop()

        except Exception as e:
            logger.error(f'Server failed to start: {e}')
        finally:
            self.cleanup()

    def handle_signal(self, signum, frame):
        """Handle various signals."""
        _ = frame  # Suppress unused parameter warnings

        if self._shutdown_requested:
            return

        match signum:
            case signal.SIGTERM | signal.SIGINT:
                logger.debug(f'Received SIG-{signum}, shutting down gracefully...')
                self._shutdown_requested = True
                if self.daemon:
                    self.daemon.shutdown()
            case signal.SIGHUP:
                logger.debug('Received SIGHUP, ignoring (restart not supported)')

    def stop(self):
        """Stop the running server."""
        pid = self._read_pid_file()
        if not pid:
            print_err('Pixa service is not running')
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
        if isinstance(self.service, RPCService):
            for name, db in self.service.databases.items():
                try:
                    db.save()
                    logger.debug(f'Database "{name}" saved')
                except Exception as e:
                    logger.error(f'Failed to save db "{name}": {e}')

        # Cleanup socket
        if UNIX_SOCKET.exists():
            logger.debug('Cleaning up socket')
            UNIX_SOCKET.unlink()

        # Remove pid file
        self._remove_pid_file()

    @classmethod
    def status(cls):
        """Check server status."""
        pid_file = BASE_DIR / 'pixa.pid'
        if not pid_file.exists():
            print_err('Pixa service is not running')
            return False

        try:
            pid = int(pid_file.read_text().strip())
            if cls.is_running(pid):
                print(f'Pixa service is running (PID: {pid})')
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
