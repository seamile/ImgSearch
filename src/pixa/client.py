import sys
from argparse import ArgumentParser
from pathlib import Path

import Pyro5.api
import Pyro5.errors
from PIL import Image

from pixa.consts import BASE_DIR, DB_NAME, SERVICE_NAME, UNIX_SOCKET
from pixa.utils import find_all_images, img2bytes, is_image, print_err, print_warn


class Client:
    """
    Pixa client for interacting with the pixa service.

    This client provides a command-line interface to the pixa service,
    supporting image search, database management, and service control operations.
    """

    def __init__(self, base_dir: Path = BASE_DIR, db_name: str = DB_NAME) -> None:
        """Initialize the pixa client."""
        self.base_dir = base_dir
        self.db_name = db_name

    def connect_to_service(self) -> Pyro5.api.Proxy:
        """Connect to the Pyro5 service via UDS and return the proxy object."""
        if not UNIX_SOCKET.exists():
            print_err(f"Service not running or socket file missing at '{UNIX_SOCKET}'.")
            print_err('You can start the service with: pixa -s start')
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
            print_err('Is the pixa service running? Check with: pixa -s status')
            sys.exit(1)
        except Exception as e:
            print_err(f'An unexpected error occurred while connecting to the service: {e}')
            sys.exit(1)

    def handle_service_command(self, service_cmd: str) -> None:
        """Handle service management commands."""
        from .server import Server

        server = Server()
        match service_cmd:
            case 'start':
                server.run()
            case 'stop':
                server.stop()
            case 'status':
                Server.status()

    def add_images(self, paths: list[str], label: str = 'path') -> None:
        """Handle adding images to the index."""
        print('Loading and sending images...')

        images_dict = {}
        for img_path in find_all_images(paths):
            try:
                label_name = img_path.stem if label == 'name' else str(img_path.resolve())
                img = Image.open(img_path)
                images_dict[label_name] = img2bytes(img)
            except Exception as e:
                print_err(f'Failed to load {img_path}: {e}')

        if not images_dict:
            print('No images found')
            return

        service = self.connect_to_service()
        queued_count: int = service.handle_add_images(images_dict)  # type: ignore
        print(f'Queued {queued_count} images for processing')

    def search(self, query: str, num: int = 10) -> None:
        """Handle search operations."""
        print('Searching...')

        service = self.connect_to_service()
        query_path = Path(query)
        if query_path.is_file() and is_image(query_path):
            # Image search
            try:
                img = Image.open(query_path)
                img_bytes = img2bytes(img)
                results: list | None = service.handle_search(img_bytes, k=num)
            except Exception as e:
                print_err(f'Failed to load image: {e}')
                return
        else:
            # Text search
            results = service.handle_search(str(query), k=num)

        if results is None:
            print('Search queue is full, please try again later.')
            return
        elif results:
            print(f'\nFound {len(results)} similar results:')
            print('-' * 60)
            for i, (path, similarity) in enumerate(results, 1):
                print(f'{i:2d}. {path} (similarity: {similarity}%)')
        else:
            print('No similar images found.')

    def get_db_info(self) -> None:
        """Handle database info request."""
        service = self.connect_to_service()
        info: dict = service.handle_get_db_info()  # type: ignore
        print('Database Information:')
        for key, value in info.items():
            print(f'  - {key.replace("_", "").title()}: {value}')

    def clear_db(self) -> None:
        """Handle database clear request."""
        service = self.connect_to_service()
        if input('Are you sure you want to clear the entire database? [y/N]: ').lower() == 'y':
            if service.handle_clear_db():
                print_warn('Database has been cleared.')
            else:
                print_err('Failed to clear the database.')
        else:
            print_warn('Operation cancelled.')

    def compare_images(self, path1: str, path2: str) -> None:
        """Handle image comparison request."""
        service = self.connect_to_service()
        abs_path1 = str(Path(path1).resolve())
        abs_path2 = str(Path(path2).resolve())
        similarity = service.handle_compare_images(abs_path1, abs_path2)
        print(f'Similarity between images: {similarity}%')


def create_parser() -> ArgumentParser:
    """Create command line argument parser."""
    parser = ArgumentParser(prog='pixa', description='Pixa - A local image search engine.')

    # Main commands
    group = parser.add_mutually_exclusive_group()
    group.add_argument('query', nargs='?', help='Search query (image path or keyword)')
    group.add_argument('-a', dest='add', nargs='+', metavar='PATH', help='Add images to DB (file or directory path)')
    group.add_argument('-i', dest='info', action='store_true', help='Show database information')
    group.add_argument('-C', dest='clear', action='store_true', help='Clear the entire database')
    group.add_argument('-c', dest='compare', nargs=2, metavar='IMG_PATH', help='Compare similarity of two images')

    # Service management commands
    srv_cmds = ['start', 'stop', 'status']
    group.add_argument('-s', dest='service', choices=srv_cmds, help='Manage the pixa service')

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

    return parser


def main() -> None:
    """Main function for command line interface"""
    parser = create_parser()
    args = parser.parse_args()

    client = Client(base_dir=args.base_dir, db_name=args.db_name)

    # Handle service management commands
    if args.service:
        client.handle_service_command(args.service)
        return

    # Handle other operations
    if args.add:
        client.add_images(args.add, args.label)
    elif args.info:
        client.get_db_info()
    elif args.clear:
        client.clear_db()
    elif args.compare:
        client.compare_images(args.compare[0], args.compare[1])
    elif args.query:
        client.search(args.query, args.num)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
