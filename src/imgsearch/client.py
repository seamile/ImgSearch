import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import Pyro5.api
import Pyro5.errors
from PIL import Image

from imgsearch import __version__
from imgsearch.consts import BASE_DIR, BATCH_SIZE, DB_NAME, DEFAULT_MODEL, SERVICE_NAME, UNIX_SOCKET
from imgsearch.utils import find_all_images, img2bytes, is_image, open_images, print_err, print_warn

Image.MAX_IMAGE_PIXELS = 100_000_000


class Client:
    """
    ImgSearch client for interacting with the imgsearch service.

    This client provides a command-line interface to the imgsearch service,
    supporting image search, database management, and service control operations.
    """

    def __init__(self, db_name: str = DB_NAME) -> None:
        """Initialize the imgsearch client."""
        self.db_name = db_name

    def connect_to_service(self) -> Pyro5.api.Proxy:
        """Connect to the Pyro5 service via UDS and return the proxy object."""
        if not UNIX_SOCKET.exists():
            print_err(f"Service not running or socket file missing at '{UNIX_SOCKET}'.")
            print_err('You can start the service with: isearch -s start')
            sys.exit(1)

        try:
            # Configure Pyro5 to use msgpack serializer
            Pyro5.config.SERIALIZER = 'msgpack'  # type: ignore

            uri = f'PYRO:{SERVICE_NAME}@./u:{UNIX_SOCKET}'
            service = Pyro5.api.Proxy(uri)
            service._pyroBind()  # A quick check to see if the server is responsive
            return service
        except Pyro5.errors.CommunicationError:
            print_err(f"Failed to connect to service socket at '{UNIX_SOCKET}'.")
            print_err('Is the imgsearch service running? Check with: isearch -s status')
            sys.exit(1)
        except Exception as e:
            print_err(f'An unexpected error occurred while connecting to the service: {e}')
            sys.exit(1)

    def handle_service_command(
        self, service_cmd: str, base_dir: Path = BASE_DIR, model_name: str = DEFAULT_MODEL
    ) -> None:
        """Handle service management commands."""
        from .server import Server

        server = Server(base_dir=base_dir, model_name=model_name)
        match service_cmd:
            case 'start':
                server.run()
            case 'stop':
                server.stop()
            case 'status':
                Server.status()

    def _preprocess_image(self, task_q: Queue, label_type: str) -> None:
        """Static method to preprocess images from queue."""
        images_dict = {}
        service = self.connect_to_service()
        try:
            while not task_q.empty():
                img_path = task_q.get(block=False)

                # convert image to bytes
                img_label = img_path.stem if label_type == 'name' else str(img_path.resolve())
                img = Image.open(img_path)
                if service.exists_in_db(img_label, self.db_name):
                    continue
                images_dict[img_label] = img2bytes(img, 480)

                # send img_bytes to server for processing
                if len(images_dict) >= BATCH_SIZE:
                    print(f'Sending {len(images_dict)} images to the server...')
                    service.handle_add_images(images_dict, self.db_name)
                    images_dict = {}
        except Exception as e:
            if not isinstance(e, Empty):
                print_err(f'Failed to preprocess image: {e}')
        finally:
            if images_dict:
                print(f'Sending {len(images_dict)} images to the server...')
                service.handle_add_images(images_dict, self.db_name)
                images_dict = {}

    def add_images(self, paths: list[str], label_type: str = 'path') -> int:
        """Handle adding images to the index using queue-based thread pool."""
        # Create queues
        task_q: Queue[Path] = Queue()

        # Find all images and add their paths to input queue
        print('Collecting images...')
        n_images = 0
        for img_path in find_all_images(paths):
            task_q.put(img_path)
            n_images += 1

        # Create fixed number of worker threads
        print('Preprocessing images...')
        threads = []
        max_workers = max(os.cpu_count() or 2, 2)
        for _ in range(max_workers):
            t = Thread(target=self._preprocess_image, args=(task_q, label_type))
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        return n_images

    def search(self, query: str, num: int = 10, similarity: float = 0.0):
        """Handle search operations."""
        query_path = Path(query)
        try:
            results: list | None
            service = self.connect_to_service()
            if query_path.is_file() and is_image(query_path):
                # Image search
                img = Image.open(query_path)
                img_bytes = img2bytes(img, 480)
                results = service.handle_search(img_bytes, k=num, similarity=similarity, db_name=self.db_name)
            else:
                # Text search
                results = service.handle_search(str(query), k=num, similarity=similarity, db_name=self.db_name)
            return results
        except Exception as e:
            print_err(f'Failed to search: {e} ({e.__class__.__name__})')
            return None

    def get_db_info(self) -> dict:
        """Handle database info request."""
        try:
            service = self.connect_to_service()
            return service.handle_get_db_info(self.db_name)  # type: ignore
        except Exception as e:
            print_err(f'Failed to get database info: {e} ({e.__class__.__name__})')
            return {}

    def clear_db(self) -> bool:
        """Handle database clear request."""
        try:
            service = self.connect_to_service()
            return service.handle_clear_db(self.db_name)  # type: ignore
        except Exception as e:
            print_err(f'Failed to clear database: {e} ({e.__class__.__name__})')
            return False

    def compare_images(self, path1: str, path2: str) -> float:
        """Handle image comparison request."""
        try:
            abs_path1 = str(Path(path1).resolve())
            abs_path2 = str(Path(path2).resolve())
            service = self.connect_to_service()
            return service.handle_compare_images(abs_path1, abs_path2)  # type: ignore
        except Exception as e:
            print_err(f'Failed to compare images: {e} ({e.__class__.__name__})')
            return 0


def create_parser() -> ArgumentParser:
    """Create command line argument parser."""
    parser = ArgumentParser(prog='imgsearch', description='ImgSearch - A local image search engine.')

    # Main commands
    group = parser.add_mutually_exclusive_group()
    group.add_argument('query', nargs='?', help='Search query (image path or keyword)')
    group.add_argument('-a', dest='add', nargs='+', metavar='PATH', help='Add images to DB (file or directory path)')
    group.add_argument('-i', dest='info', action='store_true', help='Show database information')
    group.add_argument('-C', dest='clear', action='store_true', help='Clear the entire database')
    group.add_argument('-c', dest='compare', nargs=2, metavar='IMG_PATH', help='Compare similarity of two images')

    # Service management commands
    srv_cmds = ['start', 'stop', 'status']
    group.add_argument('-s', dest='service', choices=srv_cmds, help='Manage the imgsearch service')

    # Optional arguments
    parser.add_argument(
        '--base',
        dest='base_dir',
        type=Path,
        default=BASE_DIR,
        help=f'Database base directory path (default: {BASE_DIR})',
    )
    parser.add_argument('-d', dest='db_name', type=str, default=DB_NAME, help=f'Database name (default: {DB_NAME})')
    parser.add_argument('-n', dest='num', type=int, default=10, help='Number of search results (default: 10)')
    parser.add_argument('-m', dest='model', type=str, help='CLIP model name for the service to use')
    parser.add_argument(
        '-l',
        dest='label',
        choices=['path', 'name'],
        default='path',
        help='Label naming method: "path" for absolute path, "name" for filename without extension (default: path)',
    )
    parser.add_argument(
        '-o',
        dest='open_results',
        action='store_true',
        help='Open search results with system default image viewer',
    )
    parser.add_argument(
        '--similarity',
        dest='similarity',
        type=float,
        default=0.0,
        help='Minimum similarity threshold (0-100, default: 0.0)',
    )

    group.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')

    return parser


def main() -> None:  # noqa: C901
    """Main function for command line interface"""
    parser = create_parser()
    args = parser.parse_args()

    client = Client(db_name=args.db_name)

    # Handle primary commands
    if args.service:
        client.handle_service_command(args.service, base_dir=args.base_dir, model_name=args.model or DEFAULT_MODEL)

    elif args.add:
        n_added = client.add_images(args.add, args.label)
        print(f'Added {n_added} images for processing')

    elif args.query:
        # Validate similarity parameter
        if not 0.0 <= args.similarity <= 100.0:
            print_err('Error: similarity must be between 0 and 100')
            sys.exit(1)

        print(f'Searching {args.query}...')
        results = client.search(args.query, args.num, args.similarity)
        if results:
            print(f'Found {len(results)} similar results (similarity ≥ {args.similarity}%):')
            for i, (path, similarity) in enumerate(results, 1):
                print(f'{i:2d}. {path}  {similarity}%')

            if args.open_results:
                open_images([path for path, _ in results])
        elif results is None:
            print('Search queue is full, please try again later.')
        else:
            print('No similar images found.')

    elif args.info:
        if info := client.get_db_info():
            print(f'Database "{args.db_name}":')
            for key, value in info.items():
                print(f'  - {key.title().replace("_", "")}: {value}')

    elif args.clear:
        if input(f'Are you sure to clear the database "{args.db_name}" ? [y/N]: ').lower() == 'y':
            if client.clear_db():
                print_warn(f'Database "{args.db_name}" has been cleared.')
            else:
                print_err(f'Failed to clear the database "{args.db_name}".')
        else:
            print_warn('Operation cancelled.')

    elif args.compare:
        similarity = client.compare_images(args.compare[0], args.compare[1])
        print(f'Similarity between images: {similarity}%')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
