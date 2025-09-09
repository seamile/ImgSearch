import os
import sys
from argparse import ArgumentDefaultsHelpFormatter as DefaultHelper
from argparse import ArgumentParser
from argparse import _SubParsersAction as SubParsers
from functools import cached_property
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import Pyro5.api
import Pyro5.errors
from PIL import Image

from imgsearch import __version__
from imgsearch.consts import BASE_DIR, BATCH_SIZE, DB_NAME, DEFAULT_MODEL, SERVICE_NAME, UNIX_SOCKET
from imgsearch.utils import bold, colorize, find_all_images, img2bytes, is_image, open_images, print_err, print_warn

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

    @cached_property
    def service(self) -> Pyro5.api.Proxy:
        """Connect to the Pyro5 service via UDS and return the proxy object."""
        if not UNIX_SOCKET.exists():
            print_err(f"Service not running or socket file missing at '{UNIX_SOCKET}'.")
            print_err('You can start the service with: isearch service start')
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
            print_err('Is the imgsearch service running? Check with: isearch service status')
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
        try:
            while not task_q.empty():
                img_path = task_q.get(block=False)

                # convert image to bytes
                img_label = img_path.stem if label_type == 'name' else str(img_path.resolve())
                if self.service.exists_in_db(img_label, self.db_name):
                    continue
                img = Image.open(img_path)
                images_dict[img_label] = img2bytes(img, 384)

                # send img_bytes to server for processing
                if len(images_dict) >= BATCH_SIZE:
                    print(f'Sending {len(images_dict)} images to the server...')
                    self.service.handle_add_images(images_dict, self.db_name)
                    images_dict = {}
        except Exception as e:
            if not isinstance(e, Empty):
                print_err(f'Failed to preprocess image: {e}')
        finally:
            if images_dict:
                print(f'Sending {len(images_dict)} images to the server...')
                self.service.handle_add_images(images_dict, self.db_name)
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

    def search(self, target: str, num: int = 10, similarity: int = 0):
        """Handle search operations."""
        query_path = Path(target)
        try:
            results: list | None
            if query_path.is_file() and is_image(query_path):
                # Image search
                img = Image.open(query_path)
                img_bytes = img2bytes(img, 480)
                results = self.service.handle_search(img_bytes, k=num, similarity=similarity, db_name=self.db_name)
            else:
                # Text search
                results = self.service.handle_search(str(target), k=num, similarity=similarity, db_name=self.db_name)
            return results
        except Exception as e:
            print_err(f'Failed to search: {e} ({e.__class__.__name__})')
            return None

    def list_dbs(self) -> list[str]:
        """Handle database list request."""
        try:
            return self.service.handle_list_dbs()  # type: ignore
        except Exception as e:
            print_err(f'Failed to list databases: {e} ({e.__class__.__name__})')
            return []

    def get_db_info(self) -> dict:
        """Handle database info request."""
        try:
            return self.service.handle_get_db_info(self.db_name)  # type: ignore
        except Exception as e:
            print_err(f'Failed to get database info: {e} ({e.__class__.__name__})')
            return {}

    def clear_db(self) -> bool:
        """Handle database clear request."""
        try:
            return self.service.handle_clear_db(self.db_name)  # type: ignore
        except Exception as e:
            print_err(f'Failed to clear database: {e} ({e.__class__.__name__})')
            return False

    def compare_images(self, path1: str, path2: str) -> float:
        """Handle image comparison request."""
        try:
            abs_path1 = str(Path(path1).resolve())
            abs_path2 = str(Path(path2).resolve())
            return self.service.handle_compare_images(abs_path1, abs_path2)  # type: ignore
        except Exception as e:
            print_err(f'Failed to compare images: {e} ({e.__class__.__name__})')
            return 0


def shortcut_search(parser: ArgumentParser) -> set[str]:
    """Insert search shortcut command if not provided"""
    options: set[str] = set()
    # get options
    for a in parser._actions:
        if isinstance(a, SubParsers):
            options.update(a.choices.keys())
        else:
            options.update(a.option_strings)

    # insert search command
    if len(sys.argv) > 1 and not set(sys.argv[1:]) & options:
        sys.argv.insert(1, 'search')

    return options


def create_parser() -> ArgumentParser:
    """Create command line argument parser."""
    common = ArgumentParser(add_help=False)
    common.add_argument('-d', dest='db_name', type=str, default=DB_NAME, help='Database name')

    parser = ArgumentParser(prog='isearch', description=bold('Lightweight Image Search Engine'))

    # Create subparsers for subcommands
    subcmd = parser.add_subparsers(dest='command')

    # Search Image
    cmd_search = subcmd.add_parser('search', parents=[common], help=f'Search images {bold("(default)")}')
    cmd_search.add_argument(
        '-n', dest='num', type=int, default=10, help='Number of search results (default: %(default)s)'
    )
    cmd_search.add_argument('-o', dest='open_res', action='store_true', help='Open the searched images')
    cmd_search.add_argument(
        '-m',
        dest='min_similarity',
        type=int,
        default=0,
        help='Min similarity threshold, 0 - 100 (default: %(default)s)',
    )
    cmd_search.add_argument('target', nargs='?', help='Search target (image path or keyword)')

    # Service management subcommand
    cmd_service = subcmd.add_parser('service', help='Manage the imgsearch service', formatter_class=DefaultHelper)
    cmd_service.add_argument('-b', dest='base_dir', type=Path, default=BASE_DIR, help='Database base directory path')
    cmd_service.add_argument('-m', dest='model', type=str, help='CLIP model name for the service to use')
    cmd_service.add_argument('action', choices=['start', 'stop', 'status', 'setup'], help='Service action to perform')

    # Add subcommand
    cmd_add = subcmd.add_parser('add', parents=[common], help='Add images to database', formatter_class=DefaultHelper)
    cmd_add.add_argument(
        '-l', dest='label', choices=['path', 'name'], default='path', help='Label naming method: path, name'
    )
    cmd_add.add_argument('paths', nargs='+', metavar='PATH', help='Add images to DB (file or directory path)')

    # Database management subcommand
    cmd_db = subcmd.add_parser('db', parents=[common], help='Database management operations')
    db_group = cmd_db.add_mutually_exclusive_group(required=True)
    db_group.add_argument('-i', '--info', action='store_true', help='Show database information')
    db_group.add_argument('-c', '--clear', action='store_true', help='Clear the entire database')
    db_group.add_argument('-l', '--list', action='store_true', help='List all available databases')

    # Compare subcommand
    cmd_cmp = subcmd.add_parser('cmp', help='Compare similarity of two images', formatter_class=DefaultHelper)
    cmd_cmp.add_argument('path1', metavar='IMG_PATH1', help='First image path')
    cmd_cmp.add_argument('path2', metavar='IMG_PATH2', help='Second image path')

    # Version
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')

    return parser


def main() -> None:  # noqa: C901
    """Main function for command line interface"""
    parser = create_parser()
    shortcut_search(parser)
    args = parser.parse_args()

    # Handle service subcommand
    if args.command == 'service':
        client = Client()
        client.handle_service_command(args.action, base_dir=args.base_dir, model_name=args.model or DEFAULT_MODEL)

    elif args.command == 'add':
        client = Client(db_name=args.db_name)
        n_added = client.add_images(args.paths, args.label)
        print(f'Added {n_added} images for processing')

    elif args.command == 'cmp':
        client = Client()
        similarity = client.compare_images(args.path1, args.path2)
        print(f'Similarity between images: {similarity}%')

    elif args.command == 'db':
        client = Client(db_name=args.db_name)
        if args.list:
            if databases := client.list_dbs():
                print(colorize('Available databases:', 'blue', True))
                for db_name in databases:
                    print(f'  - {db_name}')
            else:
                print_warn('No databases found.')
        elif args.info:
            if info := client.get_db_info():
                print(colorize(f'Database "{args.db_name}"', 'blue', True))
                for key, value in info.items():
                    print(f'  - {bold(key.title().replace("_", ""))}: {value}')
            else:
                print_err(f'Failed to get database info for "{args.db_name}".')
        elif args.clear:
            notice = colorize(f'Are you sure to clear the database "{args.db_name}"? [y/N]: ', 'yellow', True)
            if input(notice).lower() == 'y':
                if client.clear_db():
                    print(f'Database "{args.db_name}" has been cleared.')
                else:
                    print_err(f'Failed to clear the database "{args.db_name}".')

    elif args.command == 'search':
        client = Client(db_name=args.db_name)
        # Validate similarity parameter
        if not 0.0 <= args.min_similarity <= 100.0:
            print_err('Error: min_similarity must be between 0 and 100')
            sys.exit(1)

        print(f'Searching {args.target}...')
        results = client.search(args.target, args.num, args.min_similarity)
        if results:
            print(f'Found {len(results)} similar results (similarity ≥ {args.min_similarity}%):')
            for i, (path, similarity) in enumerate(results, 1):
                print(f'{i:2d}. {path}  {similarity}%')

            if args.open_res:
                open_images([path for path, _ in results])
        elif results is None:
            print('Search queue is full, please try again later.')
        else:
            print('No similar images found.')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
