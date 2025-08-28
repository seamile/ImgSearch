import os
import signal
import threading
from pathlib import Path
from queue import Queue
from typing import Any

import Pyro5.server
from PIL import Image

from pixa.consts import BASE_DIR, BATCH_SIZE, DB_NAME, DEFAULT_MODEL, SERVICE_NAME, UNIX_SOCKET
from pixa.storage import VectorDB
from pixa.utils import bytes2img, get_logger, print_err

BASE_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger('PixaService')


@Pyro5.server.expose
class RPCService:
    """
    RPC Service for image search operations

    This service provides a Pyro5-based RPC interface for image search operations,
    including adding images to the index, searching by image or text, and comparing images.
    All methods are thread-safe using a simple lock.
    """

    def __init__(self, db_name: str = DB_NAME, base_dir: Path = BASE_DIR, model_name: str = DEFAULT_MODEL):
        """
        Initialize the RPC service with async processing.

        Args:
            db_name: Name of the database
            base_dir: Path to the database base directory
            model_name: Name of the CLIP model to use
        """
        from pixa.clip import Clip  # import when needed

        self.clip = Clip(model_name)
        self.db = VectorDB(db_name, base_dir)
        self._lock = threading.Lock()

        # Async processing queue
        self.image_queue: Queue[tuple[str, Image.Image]] = Queue(maxsize=BATCH_SIZE * 5)
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

        # Search concurrency control
        self.max_concurrent_searches = max(2, BATCH_SIZE // 2)
        self.search_semaphore = threading.Semaphore(self.max_concurrent_searches)

    def _process_images_async(self, images: list[Image.Image], labels: list[str]):
        """Process a batch of images asynchronously."""
        try:
            features = self.clip.embed_images(images)

            with self._lock:
                self.db.add_items(labels, features)
                self.db.save()

            logger.info(f'Processed {len(images)} images')

        except Exception as e:
            logger.error(f'Failed to process batch: {e}')

    def _process_queue(self):
        """Background thread to process images from queue."""
        batch_images = []
        batch_labels = []

        while True:
            try:
                # Get image from queue
                label, image = self.image_queue.get(timeout=1)

                # Check for duplicates
                if self.db.has_label(label):
                    logger.debug(f'Skipping duplicate label: {label}')
                    continue

                batch_images.append(image)
                batch_labels.append(label)

                # Process batch when full
                if len(batch_images) >= BATCH_SIZE:
                    self._process_images_async(batch_images, batch_labels)
                    batch_images.clear()
                    batch_labels.clear()

            except Exception:
                # Process remaining images in batch
                if batch_images:
                    self._process_images_async(batch_images, batch_labels)
                    batch_images.clear()
                    batch_labels.clear()

    def handle_add_images(self, images: dict[str, bytes]) -> int:
        """
        Add images to the search index.

        Args:
            images: Dictionary mapping labels to image bytes

        Returns:
            Number of images queued for processing
        """
        logger.info(f'[AddImages] {len(images)} images received')

        queued_count = 0
        for label, image_bytes in images.items():
            try:
                image = bytes2img(image_bytes)
                self.image_queue.put((label, image))
                queued_count += 1
            except Exception:
                logger.warning(f'Queue full, dropping image: {label}')
                break

        logger.info(f'Queued {queued_count} images for processing')
        return queued_count

    def handle_search(self, query: Any, k: int = 10) -> list[tuple[str, float]] | None:
        """
        Search for similar images using image or text query.

        Args:
            query: Text description or image bytes to search for
            k: Number of results to return

        Returns:
            List of tuples containing (image_path, similarity_percentage)
            Returns None if search queue is full
            Returns empty list if no results found
        """
        # Non-blocking concurrency control
        if not self.search_semaphore.acquire(blocking=False):
            logger.warning(f'[Search] Rejected - {self.max_concurrent_searches} concurrent searches active')
            return None

        try:
            logger.info(f'[Search] type={type(query).__name__}, k={k}')

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

            return self.db.search(feature, k)
        except Exception as e:
            logger.error(f'Text search failed: {e}')
            return []
        finally:
            self.search_semaphore.release()

    def handle_get_db_info(self) -> dict:
        """
        Get database information.

        Returns:
            Dictionary containing database statistics (base_dir, size, capacity)
        """
        return {
            'base_dir': self.db.db_path.parent.as_posix(),
            'Database': self.db.db_path.name,
            'size': self.db.size,
            'capacity': self.db.capacity,
        }

    def handle_clear_db(self) -> bool:
        """
        Clear all images from the database.

        Returns:
            True if successful, False otherwise
        """
        logger.warning('[Clear] Clearing database...')
        with self._lock:
            self.db.clear()
            logger.warning('Database cleared')
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

    def __init__(self):
        self.daemon = None
        self._shutdown_requested = False
        self._restart_requested = False
        self.service = None
        self.pid_file = BASE_DIR / 'pixa.pid'

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
                logger.info('PID file removed')
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
            self.daemon = Pyro5.server.Daemon(unixsocket=str(UNIX_SOCKET))
            self.service = RPCService()
            self.daemon.register(self.service, objectId=SERVICE_NAME)

            # log service info
            logger.info(f'Listening on: {UNIX_SOCKET}')
            logger.info(f'Serializer: {Pyro5.config.SERIALIZER}')
            logger.info(f'PID: {os.getpid()}')
            logger.info(f'DB path: {self.service.db.db_path}')
            logger.info(f'DB size: {self.service.db.size}')
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
                logger.warning(f'Received SIG-{signum}, shutting down gracefully...')
                self._shutdown_requested = True
                if self.daemon:
                    self.daemon.shutdown()
            case signal.SIGHUP:
                logger.warning('Received SIGHUP, ignoring (restart not supported)')

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
        # Save database before exit
        if self.service and hasattr(self.service, 'db'):
            try:
                self.service.db.save()
                logger.info('Database saved successfully')
            except Exception as e:
                logger.error(f'Failed to save database: {e}')

        # Cleanup socket
        if UNIX_SOCKET.exists():
            logger.info('Cleaning up socket')
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
