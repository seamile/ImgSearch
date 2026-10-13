import logging
import os
import re
import signal
import stat
import threading
from collections import defaultdict
from functools import cached_property
from gc import collect
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import psutil
import Pyro5.server
from PIL import Image

from imgsearch import config as cfg
from imgsearch.models import ImageInfo, SearchResult
from imgsearch.storage import VectorDB
from imgsearch.utils import bytes2img, get_logger, img2bytes, print_err, print_warn

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
        self.logger = logger or get_logger('ImgSearchService', logging.INFO)

        self.databases: dict[str, VectorDB] = {}
        self.database_lock = threading.RLock()
        self.image_queue: Queue[tuple[str, Image.Image, str, bool, str]] = Queue(maxsize=cfg.BATCH_SIZE * 3)
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

    def _get_db(self, db_name: str):
        """Get database instance"""
        with self.database_lock:
            if db_name not in self.databases:
                self.logger.debug(f'Loading database: {db_name}')
                dim = cfg.MODELS[self.model_key][2]
                self.databases[db_name] = VectorDB(db_name, self.base_dir, dim)
            return self.databases[db_name]

    def handle_status(self) -> dict:
        """Get service status, including physical memory usage (bytes)"""
        pid = os.getpid()
        mem = psutil.Process(pid).memory_info().rss
        return {
            'PID': pid,
            'Memory': mem,
            'Base': str(self.base_dir),
            'Model': self.model_key,
        }

    def _process_images(
        self, images: list[Image.Image], paths: list[str], db_name: str,
        update_label: bool = False, dedup_mode: str = 'strict',
    ):
        """Embed queued images and merge records by vector identity."""
        try:
            if dedup_mode == 'fast':
                # Earlier queued batches may have added labels since the client checked.
                seen = set(self.handle_existing_labels([Path(path).stem for path in paths], db_name))
                pending = []
                for image, path in zip(images, paths, strict=True):
                    label = Path(path).stem
                    if label not in seen:
                        pending.append((image, path))
                        seen.add(label)
                if not pending:
                    return
                images = [image for image, _ in pending]
                paths = [path for _, path in pending]
            features = self.clip.embed_images(images)
            with self.database_lock:
                db = self._get_db(db_name)
                rows = [db.model.from_path(feature, path) for path, feature in zip(paths, features, strict=True)]
                changed = db.add_items(rows, update_label=update_label)
            self.logger.debug(f'Updated {changed} records in table "{db_name}"')
        except Exception:
            self.logger.exception('Failed to process image batch')

    def _process_queue(self) -> None:
        """Process queued images in batches grouped by table and import policy."""
        batches = defaultdict(lambda: ([], []))
        while True:
            try:
                path, image, db_name, update_label, dedup_mode = self.image_queue.get(timeout=1)
                key = (db_name, update_label, dedup_mode)
                images, paths = batches[key]
                images.append(image)
                paths.append(path)
                if len(images) >= cfg.BATCH_SIZE:
                    self._process_images(images, paths, *key)
                    del batches[key]
            except Empty:
                for key, (images, paths) in batches.items():
                    self._process_images(images, paths, *key)
                batches.clear()

    def handle_existing_labels(self, labels: list[str], db_name: str) -> list[str]:
        """Check a batch of incoming labels before the client reads image files."""
        with self.database_lock:
            return sorted(self._get_db(db_name).existing_labels(labels))

    def handle_add_images(
        self, images: dict[str, bytes], db_name: str,
        update_label: bool = False, dedup_mode: str = 'fast',
    ) -> int:
        """Queue image bytes keyed by absolute server-local paths; return accepted count."""
        if dedup_mode not in {'fast', 'strict'}:
            raise ValueError('dedup_mode must be fast or strict')
        self._get_db(db_name)  # Reject legacy tables before accepting asynchronous work.
        if any(not isinstance(path, str) or not Path(path).is_absolute() for path in images):
            raise ValueError('Image paths must be absolute server-local paths')
        queued_count = 0
        for path, image_bytes in images.items():
            try:
                image = bytes2img(image_bytes)
                self.image_queue.put((path, image, db_name, update_label, dedup_mode), timeout=5)
                queued_count += 1
            except (Full, OSError, ValueError):
                self.logger.exception('Failed to queue image %s', path)
        return queued_count

    def handle_search(
        self,
        query: Any,
        k: int = 10,
        similarity: int = 0,
        db_name: str = cfg.DB_NAME,
    ) -> list[SearchResult] | None:
        """
        Search for similar images using image or text query.

        Args:
            query: Text description or image bytes to search for
            k: Number of results to return
            similarity: Minimum similarity threshold (0-100)
            db_name: Name of the database to search in

        Returns:
            Records containing id, label, paths, and similarity score
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

            with self.database_lock:
                db = self._get_db(db_name)
                return db.search(feature, k, similarity)
        except Exception as e:
            self.logger.error(f'Search failed: {e}')
            return []
        finally:
            self.search_semaphore.release()

    def handle_list_dbs(self) -> dict[str, bool]:
        """
        List all available database names and whether they are loaded.

        Returns:
            Dictionary mapping database names to boolean indicating whether they are loaded
        """
        try:
            dbs = {name: name in self.databases for name in VectorDB.db_list(self.base_dir)}
            self.logger.debug(f'List dbs: {dbs}')
            return dbs
        except Exception as e:
            self.logger.error(f'Failed to list databases: {e}')
            return {}

    def handle_get_db_info(self, db_name: str) -> dict | None:
        """
        Get database information.

        Args:
            db_name: Name of the database to get info for

        Returns:
            Dictionary containing database statistics (base_dir, size, count)
            Returns None if database not found
        """
        with self.database_lock:
            if db_name not in VectorDB.db_list(self.base_dir):
                return None

            self.logger.debug(f'Get db info: {db_name}')
            db = self._get_db(db_name)
            return {
                'base': str(db.base.resolve()),
                'count': db.count,
                'size': db.size,
            }

    def handle_unload_dbs(self, *names: str) -> int:
        """Unload databases to free memory"""
        with self.database_lock:
            names = names or tuple(self.databases.keys())
            count = 0
            for name in names:
                if name not in self.databases:
                    continue
                self.databases.pop(name)
                count += 1
                self.logger.debug(f'Unloaded db: {name}')
        collect()
        return count

    def handle_get_images(self, keys: list[str], db_name: str, by_id: bool = False) -> list[ImageInfo]:
        """Resolve all requested labels or IDs for a deletion preview."""
        with self.database_lock:
            if db_name not in VectorDB.db_list(self.base_dir):
                raise KeyError(f'Table not found: {db_name}')
            return self._get_db(db_name).get_info(keys, by_id=by_id)

    def handle_prune_db(self, db_name: str = cfg.DB_NAME) -> dict[str, int]:  # ruff: ignore[complex-structure]
        """Clean path associations; preserve records even when every path is removed."""
        stats = {
            'checked_paths': 0,
            'deduplicated_paths': 0,
            'removed_paths': 0,
            'recomputed_paths': 0,
            'inserted_rows': 0,
            'errors': 0,
        }
        with self.database_lock:
            if db_name not in VectorDB.db_list(self.base_dir):
                raise KeyError(f'Table not found: {db_name}')
            db = self._get_db(db_name)
            with db.wlock:
                paths_by_id, changed, owners = {}, set(), defaultdict(set)
                for row in db.iter_paths():
                    paths = list(dict.fromkeys(row['paths']))
                    paths_by_id[row['id']] = paths
                    if paths != row['paths']:
                        changed.add(row['id'])
                        stats['deduplicated_paths'] += len(row['paths']) - len(paths)
                    for path in paths:
                        owners[path].add(row['id'])
                duplicates = {}
                for path, ids in owners.items():
                    stats['checked_paths'] += 1
                    try:
                        valid = stat.S_ISREG(Path(path).stat().st_mode)
                    except (FileNotFoundError, NotADirectoryError):
                        valid = False
                    except OSError:
                        stats['errors'] += 1
                        self.logger.exception('Cannot check path %s; preserving associations', path)
                        continue
                    if not valid:
                        for key in ids:
                            paths_by_id[key].remove(path)
                            changed.add(key)
                            stats['removed_paths'] += 1
                    elif len(ids) > 1:
                        duplicates[path] = ids

                # Bound vector loading when many files have disappeared.
                changed = list(changed)
                for offset in range(0, len(changed), 500):
                    records = db.get_items(changed[offset : offset + 500])
                    db.replace_items(record.model_copy(update={'paths': paths_by_id[record.id]}) for record in records)

                for path, ids in duplicates.items():
                    try:
                        # Match the pixels sent by add, including its lossy WebP conversion.
                        with Image.open(path) as image:
                            prepared = bytes2img(img2bytes(image, 384))
                        with prepared:
                            feature = self.clip.embed_image(prepared)
                        incoming = db.model.from_path(feature, path)
                        records = {record.id: record for record in db.get_items([*ids, incoming.id])}
                        target = records[incoming.id].merge(incoming) if incoming.id in records else incoming
                        replacements = [target]
                        removed = 0
                        for key in ids - {incoming.id}:
                            record = records[key]
                            replacements.append(
                                record.model_copy(update={'paths': [p for p in record.paths if p != path]})
                            )
                            removed += 1
                        # Publish each repaired path atomically, only after successful inference.
                        db.replace_items(replacements)
                        stats['removed_paths'] += removed
                        stats['recomputed_paths'] += 1
                        stats['inserted_rows'] += int(incoming.id not in records)
                    except Exception:
                        stats['errors'] += 1
                        self.logger.exception('Cannot repair path %s; preserving associations', path)
        return stats

    def handle_delete_images(self, keys: list[str], db_name: str, by_id: bool = False):
        """Delete images from the specified database."""
        self.logger.warning(f'[Delete] Delete {len(keys)} images from {db_name}')
        try:
            with self.database_lock:
                db = self._get_db(db_name)
                db.delete(*keys, by_id=by_id)
            return True
        except Exception as e:
            self.logger.error(f'Failed to delete images: {e}')
            return False

    def handle_clear_db(self, db_name: str) -> bool:
        """
        Clear all images from the specified database.

        Args:
            db_name: Name of the database to clear

        Returns:
            True if successful, False otherwise
        """
        self.logger.warning(f'[Clear] Clear database: {db_name}')
        try:
            with self.database_lock:
                db = self._get_db(db_name)
                db.clear()
            self.logger.debug(f'Database {db_name} cleared')
            return True
        except Exception as e:
            self.logger.error(f'Failed to clear database: {e}')
            return False

    def handle_drop_db(self, db_name: str) -> bool:
        """
        Drop the specified database.

        Args:
            db_name: Name of the database to drop

        Returns:
            True if successful, False otherwise
        """
        self.logger.warning(f'[Drop] Drop database: {db_name}')
        try:
            with self.database_lock:
                if db_name in self.databases:
                    del self.databases[db_name]
                return VectorDB.drop(db_name, self.base_dir)
        except Exception as e:
            self.logger.error(f'Failed to drop database: {e}')
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
        """Initialize the RPC service, logging, and server lifecycle state."""
        self._shutdown = threading.Event()
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

    def _write_pid_file(self) -> int:
        """Write current process ID to pid file."""
        try:
            pid = os.getpid()
            self.pid_file.write_text(str(pid))
            return pid
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

    def is_running(self) -> bool:
        """Check if process with given pid is running."""
        if pid := self._read_pid_file():
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        return False

    def run(self):
        """Run the server with proper signal handling."""
        # Configure Pyro5 to use msgpack serializer
        Pyro5.config.SERIALIZER = 'msgpack'  # type: ignore

        try:
            # Check if already running
            if not self.is_running():
                self.logger.info('Starting iSearch Service...')
            else:
                self.logger.error('Server already running')
                return

            # Register service and start daemon
            self.logger.info('Preloading CLIP model...')
            self.daemon = self.create_daemon()
            self.daemon.register(self.service, objectId=cfg.SERVICE_NAME)
            self.service.clip  # preload clip model  # ruff: ignore[useless-expression]

            # Create pid file
            pid = self._write_pid_file()

            # Setup signal handlers
            signal.signal(signal.SIGTERM, self.handle_signal)
            signal.signal(signal.SIGHUP, self.handle_signal)
            signal.signal(signal.SIGINT, self.handle_signal)

            # Log service info
            self.logger.info(f'iSearch service started ({pid=})')
            self.logger.debug(f'Listening : {self.bind}')
            self.logger.debug(f'Serializer: {Pyro5.config.SERIALIZER}')
            self.logger.debug(f'Base dir  : {self.base_dir}')
            self.logger.debug(f'Model     : {self.model_key}')

            self.daemon.requestLoop(lambda: not self._shutdown.is_set())
        except ImportError as e:
            self.logger.error(f'Missing optional dependencies: {e}')
            self.logger.error("Run `pip install 'imgsearch[all]'` for a full install.")
        except Exception as e:
            self.logger.error(f'Server failed to run: {e}')
        finally:
            self.cleanup()
            self.logger.info('iSearch service stopped')

    def handle_signal(self, signum, _):
        """Handle various signals."""
        match signum:
            case signal.SIGTERM | signal.SIGINT:
                name = signal.Signals(signum).name
                self.logger.warning(f'Received {name}, shutting down now...')
                self._shutdown.set()
            case signal.SIGHUP:
                self.logger.warning('Received SIGHUP, ignoring (restart not supported)')

    def stop(self):
        """Stop the running server."""
        pid = self._read_pid_file()
        if not pid:
            print_err('iSearch service is not running')
            return False

        if not self.is_running():
            print_err('iSearch service is not running')
            self._remove_pid_file()
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            print_warn(f'Shutting down process {pid}')
            return True
        except OSError as e:
            print_err(f'Failed to send SIGTERM: {e}')
            return False

    def cleanup(self):
        """Cleanup resources before exit."""
        # Shutdown daemon if running
        if self.daemon:
            try:
                self.daemon.shutdown()
                self.logger.debug('Daemon shutdown completed')
            except Exception as e:
                self.logger.error(f'Failed to shutdown daemon: {e}')

        # LanceDB commits writes immediately; release cached database handles.
        if isinstance(self.service, RPCService):
            with self.service.database_lock:
                self.service.databases.clear()

        # Cleanup socket
        if isinstance(self.bind, str):
            uds_path = Path(self.bind)
            if uds_path.is_socket():
                uds_path.unlink()
                self.logger.debug(f'Unix socket removed: {self.bind}')

        # Remove pid file
        self._remove_pid_file()


def main():
    """Main function to run the RPC service directly."""
    server = Server()
    server.run()


if __name__ == '__main__':
    main()
