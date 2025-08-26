import logging
import signal
import sys
import threading
from pathlib import Path

import numpy as np
import Pyro5.server
from PIL import Image

from ifinder.clip import Clip
from ifinder.daemon import Daemon
from ifinder.storage import DB_DIR, VectorDB
from ifinder.utils import Feature, find_all_images, is_image

# The name for the service and the socket file.
SERVICE_NAME = 'ifinder.service'
SOCKET_NAME = 'ifinder.sock'

# Setup logging
LOG_FILE = DB_DIR / 'ifinder.log'
DB_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)


class ReadWriteLock:
    """A simple read-write lock implementation."""

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()


@Pyro5.server.expose
class IFinderService:
    """
    iFinder Core Service

    This service provides a Pyro5-based RPC interface for image search operations,
    including adding images to the index, searching by image or text, and comparing images.
    All methods are thread-safe using read-write locks.
    """

    def __init__(self, db_path: Path = DB_DIR, model_name: str | None = None):
        """
        Initialize the iFinder service.

        Args:
            db_path: Path to the database directory
            model_name: Name of the CLIP model to use (optional)
        """
        logging.info('Initializing service...')
        self._lock = ReadWriteLock()
        self.clip = Clip(model_name) if model_name else Clip()
        self.db = VectorDB(db_path)
        logging.info(f'Database loaded from {db_path}, contains {self.db.size} images.')
        logging.info('Service ready.')

    def add_images(self, paths: list[str | Path], label_type: str = 'path', show_progress: bool = True) -> int:
        """
        Add images to the search index.

        Args:
            paths: List of file or directory paths containing images
            label_type: Label naming method ('path' for absolute path, 'name' for filename)
            show_progress: Whether to log progress information

        Returns:
            Number of images successfully added to the index
        """
        logging.info(f'Received request to add images from {len(paths)} paths with label_type={label_type}.')
        self._lock.acquire_write()
        try:
            image_paths = list(find_all_images(paths))
            if not image_paths:
                logging.warning('No image files found in the provided paths.')
                return 0

            # Get existing labels to avoid duplicates
            existing_labels = set(self.db.mapping.values())

            added_count = 0
            batch_size = 100

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]
                batch_images = []
                valid_labels = []

                for img_path in batch_paths:
                    try:
                        # Generate label based on label_type
                        if label_type == 'name':
                            label = Path(img_path).stem  # filename without extension
                        else:  # label_type == 'path'
                            label = str(Path(img_path).resolve())  # absolute path

                        # Skip if label already exists in database
                        if label in existing_labels:
                            logging.debug(f'Skipping {img_path}: label "{label}" already exists in database')
                            continue

                        img = Image.open(img_path).convert('RGB')
                        batch_images.append(img)
                        valid_labels.append(label)
                        existing_labels.add(label)  # Add to set to avoid duplicates within this batch
                    except Exception as e:
                        logging.error(f'Failed to load image {img_path}: {e}')
                        continue

                if not batch_images:
                    continue

                try:
                    features = self.clip.embed_images(batch_images)
                    self.db.add_items(valid_labels, features)
                    added_count += len(valid_labels)

                    if show_progress:
                        logging.info(
                            f'Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images, added {len(valid_labels)} new images'
                        )
                except Exception as e:
                    logging.error(f'Failed to process image batch: {e}')
                    continue

            if added_count > 0:
                self.db.save()
                logging.info(f'Successfully added {added_count} images.')
            return added_count
        finally:
            self._lock.release_write()

    def search(self, query: str | Path, k: int = 10) -> list[tuple[str, float]]:
        """
        Search for similar images using image or text query.

        Args:
            query: Image path or text description to search for
            k: Number of results to return

        Returns:
            List of tuples containing (image_path, similarity_percentage)
        """
        logging.info(f'Received search query: "{query}"')
        feature: Feature | None = None

        self._lock.acquire_read()
        try:
            query_path = Path(query)
            if query_path.exists() and is_image(query_path):
                try:
                    img = Image.open(query_path).convert('RGB')
                    feature = self.clip.embed_image(img)
                except Exception as e:
                    logging.error(f'Image search failed on loading: {e}')
                    return []
            else:
                try:
                    feature = self.clip.embed_text(str(query))
                except Exception as e:
                    logging.error(f'Text search failed on embedding: {e}')
                    return []

            if feature:
                return self.db.search(feature, k)
            return []
        finally:
            self._lock.release_read()

    def get_db_info(self) -> dict:
        """
        Get database information.

        Returns:
            Dictionary containing database statistics (base_dir, size, capacity)
        """
        self._lock.acquire_read()
        try:
            return {
                'base_dir': str(self.db.base_dir),
                'size': self.db.size,
                'capacity': self.db.capacity,
            }
        finally:
            self._lock.release_read()

    def clear_db(self) -> bool:
        """
        Clear all images from the database.

        Returns:
            True if successful, False otherwise
        """
        logging.info('Received request to clear database.')
        self._lock.acquire_write()
        try:
            self.db.clear()
            logging.info('Database cleared.')
            return True
        except Exception as e:
            logging.error(f'Failed to clear database: {e}')
            return False
        finally:
            self._lock.release_write()

    def compare_images(self, path1: str, path2: str) -> float:
        """
        Compare similarity between two images.

        Args:
            path1: Path to the first image
            path2: Path to the second image

        Returns:
            Similarity percentage (0-100)
        """
        logging.info(f'Received request to compare {path1} and {path2}')
        self._lock.acquire_read()
        try:
            img1 = Image.open(path1).convert('RGB')
            img2 = Image.open(path2).convert('RGB')
            feature1 = self.clip.embed_image(img1)
            feature2 = self.clip.embed_image(img2)

            similarity = np.dot(feature1, feature2)
            return round(float(similarity) * 100, 2)
        except Exception as e:
            logging.error(f'Failed to compare images: {e}')
            return 0.0
        finally:
            self._lock.release_read()


class ServerDaemon(Daemon):
    """Daemon for the iFinder service."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon = None
        self._shutdown_requested = False

    def run(self):
        uds_socket = DB_DIR / SOCKET_NAME
        if uds_socket.exists():
            uds_socket.unlink()

        try:
            self.daemon = Pyro5.server.Daemon(unixsocket=str(uds_socket))
            service = IFinderService()
            self.daemon.register(service, objectId=SERVICE_NAME)
            signal.signal(signal.SIGTERM, self.handle_shutdown)
            logging.info(f'iFinder service started. Listening on unix socket: {uds_socket}')
            self.daemon.requestLoop()
        except Exception as e:
            logging.error(f'Server failed to start: {e}')
        finally:
            self.cleanup()

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signal to avoid infinite loop."""
        _ = signum, frame  # Suppress unused parameter warnings
        if self._shutdown_requested:
            return  # Avoid handling multiple shutdown signals

        self._shutdown_requested = True
        logging.info('Received shutdown signal. Shutting down gracefully...')

        if self.daemon:
            try:
                self.daemon.shutdown()
            except Exception as e:
                logging.error(f'Error during daemon shutdown: {e}')

    def cleanup(self):
        uds_socket = DB_DIR / SOCKET_NAME
        if uds_socket.exists():
            logging.info('Shutting down and cleaning up socket.')
            uds_socket.unlink()


def main():
    # This main function is for manual, non-daemonized testing.
    daemon = ServerDaemon()
    daemon.run()


if __name__ == '__main__':
    main()
