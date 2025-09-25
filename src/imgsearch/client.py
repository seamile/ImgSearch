import re
import sys
from argparse import ArgumentDefaultsHelpFormatter as DefaultFmt
from argparse import ArgumentParser
from argparse import _SubParsersAction as SubParsers
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import Pyro5.api
import Pyro5.errors
from PIL import Image

from imgsearch import __version__
from imgsearch import config as cfg
from imgsearch import utils as ut
from imgsearch.exceptions import NotRunningError
from imgsearch.server import Server
from imgsearch.setup import remove_service, setup_service

Pyro5.config.COMPRESSION = True  # type: ignore
Image.MAX_IMAGE_PIXELS = 900_000_000
NETLOC_PATTERN = re.compile(r'^([a-zA-Z0-9]+[.:])+([a-zA-Z0-9]+):[1-9]\d{3,4}$')
_thread_local = local()


class Client:
    """
    ImgSearch client for interacting with the imgsearch service.

    This client provides a command-line interface to the imgsearch service,
    supporting image search, database management, and service control operations.
    """

    def __init__(self, db_name: str = cfg.DB_NAME, bind: str = cfg.UNIX_SOCKET) -> None:
        """Initialize the imgsearch client."""
        self.db_name = db_name
        self.bind = bind

    @property
    def service(self) -> Pyro5.api.Proxy:
        """Connect to the Pyro5 service via UDS and return the proxy object."""
        if not hasattr(_thread_local, 'service'):
            if NETLOC_PATTERN.match(self.bind):
                # Assume ip:port format
                host, port = self.bind.split(':', 1)
                uri = f'PYRO:{cfg.SERVICE_NAME}@{host}:{port}'
            elif Path(self.bind).is_socket():
                uri = f'PYRO:{cfg.SERVICE_NAME}@./u:{self.bind}'
            else:
                raise NotRunningError(f'Service not running or socket file missing at {self.bind}.')

            try:
                # Configure Pyro5 to use msgpack serializer
                Pyro5.config.SERIALIZER = 'msgpack'  # type: ignore
                _thread_local.service = Pyro5.api.Proxy(uri)
                _thread_local.service._pyroBind()  # A quick check to see if the server is responsive
            except Pyro5.errors.CommunicationError as e:
                raise NotRunningError(f'Service not running or socket file missing at {self.bind}.') from e

        return _thread_local.service

    def service_status(self) -> dict:
        """Get the status of the imgsearch service."""
        return self.service.handle_status()  # type: ignore

    def _preprocess_images(self, batch: dict[str, str]) -> None:
        """Process a batch of image paths."""
        send_dict: dict[str, bytes] = {}
        for label, path in batch.items():
            try:
                img = Image.open(path)
                send_dict[label] = ut.img2bytes(img, 384)
                if len(send_dict) >= cfg.BATCH_SIZE:
                    ut.print_inf(f'Sending {len(send_dict)} images to the server...')
                    self.service.handle_add_images(send_dict, self.db_name)
                    send_dict = {}
            except Exception as e:
                ut.print_err(f'Failed to process image {path}: {e}')

        if send_dict:
            ut.print_inf(f'Sending {len(send_dict)} images to the server...')
            self.service.handle_add_images(send_dict, self.db_name)

    def _filter_out_exists(self, imgs: dict[str, str]) -> dict[str, str]:
        """Filter out existing images from the database."""
        if labels := list(imgs.keys()):
            result: list[bool] = self.service.handle_check_exist_labels(labels, self.db_name)  # type: ignore
            return {lb: imgs[lb] for lb, exist in zip(labels, result, strict=True) if not exist}
        return {}

    def add_images(self, paths: list[str], label_type: str = 'path') -> int:
        """Handle adding images to the index using thread pool."""
        ut.print_inf('Collecting images...')
        pool = ThreadPoolExecutor(max_workers=max(ut.cpu_count(), 2))
        found_ipaths: dict[str, str] = {}
        to_added: dict[str, str] = {}
        n_images = 0

        for img_path in ut.find_all_images(paths):
            ut.print_msg(f'Found {img_path}')
            label = img_path.stem if label_type == 'name' else str(img_path.resolve())
            found_ipaths[label] = str(img_path)

            if len(found_ipaths) >= cfg.BATCH_SIZE:
                new_images = self._filter_out_exists(found_ipaths)
                to_added.update(new_images)
                found_ipaths = {}
                if len(to_added) >= cfg.BATCH_SIZE:
                    pool.submit(self._preprocess_images, to_added)
                    n_images += len(to_added)
                    to_added = {}

        # Process remaining found_ipaths
        if found_ipaths:
            new_images = self._filter_out_exists(found_ipaths)
            to_added.update(new_images)

        # Submit remaining to_added
        if to_added:
            pool.submit(self._preprocess_images, to_added)
            n_images += len(to_added)

        ut.print_inf(f'Preprocessing {n_images} images...')
        pool.shutdown(wait=True)

        return n_images

    def search(self, target: str, num: int = 10, similarity: int = 0):
        """Handle search operations."""
        query_path = Path(target)
        try:
            results: list | None
            if query_path.is_file() and ut.is_image(query_path):
                # Image search
                img = Image.open(query_path)
                img_bytes = ut.img2bytes(img, 384)
                results = self.service.handle_search(img_bytes, k=num, similarity=similarity, db_name=self.db_name)
            else:
                # Text search
                results = self.service.handle_search(str(target), k=num, similarity=similarity, db_name=self.db_name)
            return results
        except Exception as e:
            ut.print_err(f'Failed to search: {e} ({e.__class__.__name__})')
            return None

    def list_dbs(self) -> list[str]:
        """Handle database list request."""
        try:
            return self.service.handle_list_dbs()  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return []

    def get_db_info(self) -> dict | None:
        """Handle database info request."""
        try:
            return self.service.handle_get_db_info(self.db_name)
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return None

    def delete_images(self, labels: list[str], rebuild: bool = False) -> bool:
        """Handle image deletion request."""
        try:
            return self.service.handle_delete_images(labels, rebuild, self.db_name)  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return False

    def clear_db(self) -> bool:
        """Handle database clear request."""
        try:
            return self.service.handle_clear_db(self.db_name)  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return False

    def drop_db(self) -> bool:
        """Handle database drop request."""
        try:
            return self.service.handle_drop_db(self.db_name)  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return False

    def compare_images(self, path1: str, path2: str) -> float:
        """Handle image comparison request."""
        try:
            img1 = Image.open(path1)
            ibytes1 = ut.img2bytes(img1, 384)

            img2 = Image.open(path2)
            ibytes2 = ut.img2bytes(img2, 384)

            return self.service.handle_compare_images(ibytes1, ibytes2)  # type: ignore
        except Exception as e:
            ut.print_err(f'Failed to compare images: {e} ({e.__class__.__name__})')
            return 0


def handle_service_command(  # noqa: C901
    service_cmd: str,
    base_dir: Path = cfg.BASE_DIR,
    model_key: str = cfg.DEFAULT_MODEL_KEY,
    bind: str = cfg.UNIX_SOCKET,
    log_level: str = 'info',
) -> None:
    """Handle service management commands."""
    match service_cmd:
        case 'start' | 'stop' | 'status':
            server = Server(base_dir, model_key, bind, log_level)
            if service_cmd == 'start':
                try:
                    server.run()
                except Exception as e:
                    server.logger.error(f'{service_cmd.title()} failed: {e}')
            elif service_cmd == 'stop':
                try:
                    server.stop()
                except Exception as e:
                    server.logger.error(f'{service_cmd.title()} failed: {e}')
            else:
                if not server.is_running():
                    ut.print_err('iSearch service is not running')
                    sys.exit(1)

                try:
                    client = Client(bind=bind)
                    status: dict = client.service_status()
                    ut.print_inf('iSearch service is running', marked=True)
                    ut.print_inf(f' - {ut.bold("PID")}  : {status["PID"]}')
                    ut.print_inf(f' - {ut.bold("MEM")}  : {status["Memory"] / 1024 / 1024:.1f} MB')
                    ut.print_inf(f' - {ut.bold("Base")} : {status["Base"]}')
                    ut.print_inf(f' - {ut.bold("Model")}: {status["Model"]}')
                except NotRunningError:
                    ut.print_err('iSearch service is not running')
                    sys.exit(1)
                except Exception as e:
                    ut.print_err(f'{e.__class__.__name__}: {e}')
                    sys.exit(1)

        case 'setup' | 'remove':
            try:
                msg = ut.bold(f'Are you sure to {service_cmd} isearch service? [y/N]: ')
                if input(msg).lower() != 'y':
                    return
                elif service_cmd == 'setup' and setup_service(base_dir, model_key, bind, log_level):
                    ut.print_inf(f'Service {service_cmd} completed successfully.')
                elif service_cmd == 'remove' and remove_service():
                    ut.print_inf(f'Service {service_cmd} completed successfully.')
                else:
                    ut.print_err(f'Failed to {service_cmd} isearch service.')
                    sys.exit(1)
            except Exception as e:
                ut.print_err(f'{service_cmd.title()} failed: {e}')
                sys.exit(1)


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
    # Common arguments
    arg_bind = ArgumentParser(add_help=False)
    arg_bind.add_argument(
        '-B',
        dest='bind',
        default=cfg.UNIX_SOCKET,
        metavar='ADDR',
        help='Server bind address (UDS path or ip:port)',
    )
    arg_db = ArgumentParser(add_help=False)
    arg_db.add_argument('-d', dest='db_name', default=cfg.DB_NAME, help='Database name')

    # Main parser
    parser = ArgumentParser(prog='isearch', description=ut.bold('Lightweight Image Search Engine'))

    # Create subparsers for subcommands
    subcmd = parser.add_subparsers(dest='command')

    # Search image subcommand
    cmd_search = subcmd.add_parser(
        'search',
        parents=[arg_bind, arg_db],
        help=f'Search images {ut.bold("(default)")}',
        formatter_class=DefaultFmt,
    )
    cmd_search.add_argument('-t', dest='sim_thr', type=int, default=0, help='Similarity threshold, 0 - 100')
    cmd_search.add_argument('-n', dest='num', type=int, default=10, help='Number of search results')
    cmd_search.add_argument('-o', dest='open_res', action='store_true', help='Open the searched images')
    cmd_search.add_argument('target', nargs='?', help='Search target (image path or keyword)')

    # Service management subcommand
    cmd_service = subcmd.add_parser(
        'service',
        parents=[arg_bind],
        help='Manage the iSearch service',
        formatter_class=DefaultFmt,
    )
    cmd_service.add_argument(
        '-b',
        dest='base_dir',
        type=Path,
        default=cfg.BASE_DIR,
        help='Database base directory path',
    )
    cmd_service.add_argument(
        '-m',
        dest='model_key',
        choices=sorted(cfg.MODELS.keys()),
        default=cfg.DEFAULT_MODEL_KEY,
        metavar='MODEL_KEY',
        help='CLIP model key for the service to use, options: %(choices)s',
    )
    cmd_service.add_argument(
        'action',
        choices=['start', 'stop', 'status', 'setup', 'remove'],
        metavar='ACTION',
        help='Service action to perform, options: start, stop, status, setup, remove',
    )
    cmd_service.add_argument(
        '-L',
        dest='log_level',
        choices=['debug', 'info', 'warning', 'error'],
        default='info',
        metavar='LOG_LEVEL',
        help='Log level for the service, options: %(choices)s',
    )

    # Add images subcommand
    cmd_add = subcmd.add_parser(
        'add',
        parents=[arg_bind, arg_db],
        help='Add images to database',
        formatter_class=DefaultFmt,
    )
    cmd_add.add_argument('-l', dest='label', choices=['path', 'name'], default='path', help='Label naming method')
    cmd_add.add_argument('paths', nargs='+', metavar='PATH', help='Add images to DB (file or directory path)')

    # Database management subcommand
    cmd_db = subcmd.add_parser('db', parents=[arg_bind], help='Database management operations')
    cmd_db.add_argument('-r', '--rebuild', action='store_true', help='Rebuild index after deletion')
    cmd_db.add_argument('db_name', nargs='?', metavar='DB_NAME', help='Database name')
    db_group = cmd_db.add_mutually_exclusive_group(required=True)
    db_group.add_argument('-l', '--list', action='store_true', help='List all available databases')
    db_group.add_argument('-i', '--info', action='store_true', help='Show database information')
    db_group.add_argument('-d', '--delete', nargs='+', metavar='LABEL', help='Delete images by label')
    db_group.add_argument('-c', '--clear', action='store_true', help='Clear the entire database')
    db_group.add_argument('-D', '--drop', action='store_true', help='Drop the specified database')

    # Compare images subcommand
    cmd_cmp = subcmd.add_parser('cmp', parents=[arg_bind], help='Compare similarity of two images')
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
        handle_service_command(
            args.action,
            base_dir=args.base_dir,
            model_key=args.model_key,
            bind=args.bind,
            log_level=args.log_level,
        )

    elif args.command == 'add':
        client = Client(db_name=args.db_name, bind=args.bind)
        n_added = client.add_images(args.paths, args.label)
        ut.print_inf(f'Added {n_added} images for processing')

    elif args.command == 'cmp':
        client = Client(bind=args.bind)
        similarity = client.compare_images(args.path1, args.path2)
        ut.print_inf(f'Similarity between images: {similarity}%')

    elif args.command == 'db':
        client = Client(db_name=args.db_name, bind=args.bind)
        if args.list:
            if databases := client.list_dbs():
                ut.print_inf('Databases:', marked=True)
                for db_name in databases:
                    ut.print_inf(f' - {db_name}')
            else:
                ut.print_warn('No databases found.')

        elif args.info:
            if info := client.get_db_info():
                ut.print_inf(f'Database "{args.db_name}"', marked=True)
                for key, value in sorted(info.items()):
                    if key == 'size':
                        value = f'{value / 1024 / 1024:.1f} MB'
                    ut.print_inf(f' - {ut.bold(key.title().replace("_", ""))}: {value}')
            elif args.db_name is None:
                ut.print_err('Database name is required.')
            else:
                ut.print_err(f"Not found DB: '{args.db_name}'.")

        elif args.delete:
            if args.db_name is None:
                ut.print_err('Database name is required.')
                sys.exit(1)

            notice = ut.colorize(
                f'Are you sure to delete {len(args.delete)} images from DB "{args.db_name}"? [y/N]: ', 'yellow', True
            )
            if input(notice).lower() == 'y':
                if client.delete_images(args.delete, args.rebuild):
                    ut.print_inf(f'Successfully deleted {len(args.delete)} images from DB "{ut.bold(args.db_name)}".')
                else:
                    ut.print_err(f'Failed to delete images from DB "{args.db_name}".')

        elif args.clear:
            if args.db_name is None:
                ut.print_err('Database name is required.')
                sys.exit(1)

            notice = ut.colorize(f'Are you sure to clear the DB "{args.db_name}"? [y/N]: ', 'yellow', True)
            if input(notice).lower() == 'y':
                if client.clear_db():
                    ut.print_inf(f'Database "{ut.colorize(args.db_name, "red")}" has been cleared.')
                else:
                    ut.print_err(f'Failed to clear the DB "{args.db_name}".')

        elif args.drop:
            if args.db_name is None:
                ut.print_err('Database name is required.')
                sys.exit(1)

            notice = ut.colorize(f'Are you sure to drop the DB "{args.db_name}"? [y/N]: ', 'yellow', True)
            if input(notice).lower() == 'y':
                if client.drop_db():
                    ut.print_inf(f'Database "{ut.colorize(args.db_name, "red")}" has been dropped.')
                else:
                    ut.print_err(f'Failed to drop the DB "{args.db_name}".')

    elif args.command == 'search':
        client = Client(db_name=args.db_name, bind=args.bind)
        # Validate similarity parameter
        if not 0.0 <= args.sim_thr <= 100.0:
            ut.print_err('Error: sim_thr must be between 0 and 100')
            sys.exit(1)

        ut.print_inf(f'Searching {args.target}...')
        results = client.search(args.target, args.num, args.sim_thr)
        if results:
            ut.print_inf(f'Found {len(results)} similar images (similarity ≥ {args.sim_thr}%):')
            for i, (path, similarity) in enumerate(results, 1):
                ut.print_inf(f'{i:2d}. {path}\t{similarity}%')

            if args.open_res:
                ut.open_images([path for path, _ in results])
        elif results is None:
            ut.print_inf('Search queue is full, please try again later.')
        else:
            ut.print_inf('No similar images found.')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
