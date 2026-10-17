import os
import re
import sys
from argparse import ArgumentDefaultsHelpFormatter as DefaultFmt
from argparse import ArgumentParser
from argparse import _SubParsersAction as SubParsers
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from typing import cast

import Pyro5.api
import Pyro5.errors
from PIL import Image

from imgsearch import __version__
from imgsearch import config as cfg
from imgsearch import utils as ut
from imgsearch.exceptions import NotRunningError
from imgsearch.models import ImageInfo, SearchResult
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
    supporting image search, table management, and service control operations.
    """

    def __init__(self, table_name: str = cfg.TABLE_NAME, bind: str = cfg.UNIX_SOCKET) -> None:
        """Initialize the imgsearch client."""
        self.table_name = table_name
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

    def _preprocess_images(self, paths: list[str], update_label: bool, dedup_mode: str) -> int:
        """Encode and submit readable images, returning the accepted queue count."""
        if dedup_mode == 'fast':
            try:
                path_labels = [(path, Path(path).stem) for path in paths]
                labels = set(self.service.handle_existing_labels([label for _, label in path_labels], self.table_name))  # type: ignore
            except Exception as e:
                ut.print_err(f'Failed to check existing labels: {e}')
                return 0
            pending = [path for path, label in path_labels if label not in labels]
            if skipped := len(paths) - len(pending):
                ut.print_inf(f'Skipped {skipped} images with existing labels in "{self.table_name}"')
            paths = pending
        if not paths:
            return 0

        ut.print_inf(f'Preparing {len(paths)} images for "{self.table_name}"...')
        images = {}
        for path in paths:
            try:
                with Image.open(path) as image:
                    images[path] = ut.img2bytes(image, 384)
            except Exception as e:
                ut.print_err(f'Failed to process image {path}: {e}')
        if not images:
            return 0
        try:
            ut.print_inf(f'Submitting {len(images)} images to "{self.table_name}"...')
            accepted = cast(
                int,
                self.service.handle_add_images(
                    images,
                    self.table_name,
                    update_label=update_label,
                    dedup_mode=dedup_mode,
                ),
            )
            if accepted < len(images):
                ut.print_warn(
                    f'Server accepted {accepted}/{len(images)} images; {len(images) - accepted} were not queued'
                )
            else:
                ut.print_inf(f'Queued {accepted} images in "{self.table_name}"')
            return accepted
        except Exception as e:
            ut.print_err(f'Failed to submit images: {e}')
            return 0

    def add_images(self, paths: list[str], *, update_label: bool = False, dedup_mode: str = 'fast') -> int:
        """Skip stored labels in fast mode, or embed all images in strict mode; return accepted count."""
        if dedup_mode not in {'fast', 'strict'}:
            raise ValueError('dedup_mode must be fast or strict')
        ut.print_inf(f'Scanning for images to add to "{self.table_name}" (dedup={dedup_mode})...')
        futures, batch = [], []
        found = 0
        with ThreadPoolExecutor(max_workers=max(ut.cpu_count(), 2)) as pool:
            for path in ut.find_all_images(paths):
                found += 1
                ut.print_msg(f'Found image #{found}: {path}')
                batch.append(os.path.abspath(os.path.expanduser(str(path))))
                if len(batch) >= cfg.BATCH_SIZE:
                    futures.append(pool.submit(self._preprocess_images, batch, update_label, dedup_mode))
                    batch = []
            if batch:
                futures.append(pool.submit(self._preprocess_images, batch, update_label, dedup_mode))
            if not found:
                ut.print_warn('No image files found.')
            else:
                ut.print_inf(f'Found {found} image files; waiting for preparation and submission...')
            return sum(future.result() for future in futures)

    def search(self, target: str | Image.Image, num: int = 10, similarity: int = 0) -> list[SearchResult] | None:
        """Handle search operations."""
        query: str | bytes
        if isinstance(target, str):
            query_path = Path(target)
            if query_path.is_file() and ut.is_image(query_path):
                # search by image path
                img = Image.open(query_path)
                query = ut.img2bytes(img, 384)
            else:
                # search by text
                query = target
        elif isinstance(target, Image.Image):
            # search by image object
            query = ut.img2bytes(target, 384)
        else:
            raise TypeError(f'Invalid query type: {type(target)}')

        try:
            return self.service.handle_search(query, k=num, similarity=similarity, table_name=self.table_name)
        except Exception as e:
            ut.print_err(f'Failed to search: {e} ({e.__class__.__name__})')
            return None

    def list_tables(self) -> dict[str, bool]:
        """Handle table list request."""
        try:
            return self.service.handle_list_tables()  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return {}

    def unload_tables(self, *names: str) -> int:
        """Handle table unload request."""
        try:
            return self.service.handle_unload_tables(*names)  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return 0

    def get_table_info(self) -> dict | None:
        """Handle table info request."""
        try:
            return self.service.handle_get_table_info(self.table_name)
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return None

    def get_images(self, keys: list[str], *, by_id: bool = False) -> list[ImageInfo]:
        """Fetch image metadata by label or ID from the service."""
        return cast(list[ImageInfo], self.service.handle_get_images(keys, self.table_name, by_id=by_id))

    def prune_table(self) -> dict[str, int]:
        """Prune stale or conflicting image paths and return cleanup statistics."""
        ut.print_inf(f'Pruning table "{self.table_name or cfg.TABLE_NAME}"; checking paths and repairing conflicts...')
        return cast(dict[str, int], self.service.handle_prune_table(self.table_name or cfg.TABLE_NAME))

    def delete_images(self, keys: list[str], *, by_id: bool = False) -> bool:
        """Handle image deletion request."""
        try:
            return self.service.handle_delete_images(keys, self.table_name, by_id=by_id)  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return False

    def clear_table(self) -> bool:
        """Handle table clear request."""
        try:
            return self.service.handle_clear_table(self.table_name)  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return False

    def drop_table(self) -> bool:
        """Handle table drop request."""
        try:
            return self.service.handle_drop_table(self.table_name)  # type: ignore
        except Exception as e:
            ut.print_err(f'{e.__class__.__name__}: {e}')
            return False

    def compare_images(self, path1: str, path2: str) -> float:
        """Handle image comparison request."""
        ut.print_msg(f'Comparing {path1} and {path2}...')
        try:
            img1 = Image.open(path1)
            ibytes1 = ut.img2bytes(img1, 384)

            img2 = Image.open(path2)
            ibytes2 = ut.img2bytes(img2, 384)

            return self.service.handle_compare_images(ibytes1, ibytes2)  # type: ignore
        except Exception as e:
            ut.print_err(f'Failed to compare images: {e} ({e.__class__.__name__})')
            return 0


def handle_service_command(  # ruff: ignore[complex-structure]
    service_cmd: str,
    db_path: Path = cfg.DB_PATH,
    model_key: str = cfg.DEFAULT_MODEL_KEY,
    bind: str = cfg.UNIX_SOCKET,
    log_level: str = 'info',
) -> None:
    """Handle service management commands."""
    match service_cmd:
        case 'start' | 'stop' | 'status':
            server = Server(db_path, model_key, bind, log_level)
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
                    ut.print_inf(f' - {ut.bold("DB path")} : {status["db_path"]}')
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
                elif service_cmd == 'setup' and setup_service(db_path, model_key, bind, log_level):
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
    arg_table = ArgumentParser(add_help=False)
    arg_table.add_argument('-d', dest='table_name', default=cfg.TABLE_NAME, help='Image table name')

    # Main parser
    parser = ArgumentParser(prog='isearch', description=ut.bold('Lightweight Image Search Engine'))

    # Create subparsers for subcommands
    subcmd = parser.add_subparsers(dest='command')

    # Search image subcommand
    cmd_search = subcmd.add_parser(
        'search',
        parents=[arg_bind, arg_table],
        help=f'Search images {ut.bold("(default)")}',
        formatter_class=DefaultFmt,
    )
    cmd_search.add_argument('-t', dest='sim_thr', type=int, default=0, help='Similarity threshold, 0 - 100')
    cmd_search.add_argument('-n', dest='num', type=int, default=3, help='Number of search results')
    cmd_search.add_argument('-o', dest='open_res', action='store_true', help='Open the searched images')
    cmd_search.add_argument('target', help='Search target (image path or keyword)')

    # Service management subcommand
    cmd_service = subcmd.add_parser(
        'service',
        parents=[arg_bind],
        help='Manage the iSearch service',
        formatter_class=DefaultFmt,
    )
    cmd_service.add_argument(
        '-d',
        dest='db_path',
        type=Path,
        default=cfg.DB_PATH,
        help='LanceDB root directory path',
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
        parents=[arg_bind, arg_table],
        help='Add images to table',
        formatter_class=DefaultFmt,
    )
    cmd_add.add_argument(
        '--update-label', action='store_true', help='Update existing image labels from incoming filenames'
    )
    cmd_add.add_argument(
        '--dedup-mode',
        choices=['fast', 'strict'],
        default='fast',
        help='fast: skip existing labels; strict: embed every image and merge by vector ID',
    )
    cmd_add.add_argument('paths', nargs='+', metavar='PATH', help='Add images to table (file or directory path)')

    # Table management subcommand
    cmd_table = subcmd.add_parser('table', parents=[arg_bind], help='Manage named image tables in the LanceDB root')
    cmd_table.add_argument('table_name', nargs='?', metavar='TABLE_NAME', help='Image table name')
    table_group = cmd_table.add_mutually_exclusive_group(required=True)
    table_group.add_argument('-l', '--list', action='store_true', help='List all available image tables')
    table_group.add_argument('-i', '--info', action='store_true', help='Show selected table information')
    table_group.add_argument('-u', '--unload', action='store_true', help='Unload all image tables from memory')
    table_group.add_argument(
        '--delete', nargs='+', metavar='LABEL', help='Preview and delete all images matching labels'
    )
    table_group.add_argument('--delete-id', nargs='+', metavar='ID', help='Preview and delete images by vector ID')
    table_group.add_argument(
        '--prune', action='store_true', help='Repair paths in the selected table (default: default)'
    )
    table_group.add_argument('--clear', action='store_true', help='Clear the selected image table')
    table_group.add_argument('--drop', action='store_true', help='Drop the selected image table')

    # Compare images subcommand
    cmd_cmp = subcmd.add_parser('cmp', parents=[arg_bind], help='Compare similarity of two images')
    cmd_cmp.add_argument('path1', metavar='IMG_PATH1', help='First image path')
    cmd_cmp.add_argument('path2', metavar='IMG_PATH2', help='Second image path')

    # Version
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')

    return parser


def main() -> None:  # ruff: ignore[complex-structure]
    """Main function for command line interface"""
    parser = create_parser()
    shortcut_search(parser)
    args = parser.parse_args()

    # Handle service subcommand
    if args.command == 'service':
        handle_service_command(
            args.action,
            db_path=args.db_path,
            model_key=args.model_key,
            bind=args.bind,
            log_level=args.log_level,
        )

    elif args.command == 'add':
        client = Client(table_name=args.table_name, bind=args.bind)
        n_added = client.add_images(args.paths, update_label=args.update_label, dedup_mode=args.dedup_mode)
        ut.print_inf(f'Submitted {n_added} images for background processing; check the server log for completion')

    elif args.command == 'cmp':
        client = Client(bind=args.bind)
        similarity = client.compare_images(args.path1, args.path2)
        ut.print_inf(f'Similarity between images: {similarity}%')

    elif args.command == 'table':
        client = Client(table_name=args.table_name, bind=args.bind)
        if args.list:
            if tables := client.list_tables():
                ut.print_inf('Tables:', marked=True)
                for table_name, loaded in tables.items():
                    if loaded:
                        ut.print_inf(f' - {ut.bold(table_name)} {ut.colorize("(loaded)", "gray")}')
                    else:
                        ut.print_inf(f' - {table_name}')
            else:
                ut.print_warn('No tables found.')

        elif args.info:
            if info := client.get_table_info():
                ut.print_inf(f'Table "{args.table_name}"', marked=True)
                for key, value in sorted(info.items()):
                    if key == 'size':
                        value = f'{value / 1024 / 1024:.1f} MB'
                    ut.print_inf(f' - {ut.bold(key.title().replace("_", ""))}: {value}')
            elif args.table_name is None:
                ut.print_err('Table name is required.')
            else:
                ut.print_err(f"Table not found: '{args.table_name}'")

        elif args.unload:
            if count := client.unload_tables():
                ut.print_inf(f'Unloaded {count} tables.')
            else:
                ut.print_inf('Not loaded any tables.')

        elif args.prune:
            try:
                stats = client.prune_table()
                for name, count in stats.items():
                    print(f'{name}: {count}')
                if stats['errors']:
                    sys.exit(1)
            except Exception as e:
                ut.print_err(f'Prune failed: {e}')
                sys.exit(1)

        elif args.delete or args.delete_id:
            if args.table_name is None:
                ut.print_err('Table name is required.')
                sys.exit(1)
            try:
                rows = client.get_images(args.delete_id or args.delete, by_id=bool(args.delete_id))
            except Exception as e:
                ut.print_err(f'Cannot prepare deletion: {e}')
                sys.exit(1)
            for row in rows:
                print(f'{row["id"]}\t{row["label"]}\t{row["paths"]}')
            notice = f'Delete these {len(rows)} image records from "{args.table_name}"? [y/N]: '
            if input(notice).lower() == 'y':
                if client.delete_images([row['id'] for row in rows], by_id=True):
                    ut.print_inf(f'Successfully deleted {len(rows)} image records.')
                else:
                    ut.print_err('Failed to delete image records.')
                    sys.exit(1)

        elif args.clear:
            if args.table_name is None:
                ut.print_err('Table name is required.')
                sys.exit(1)

            notice = ut.colorize(f'Are you sure to clear the table "{args.table_name}"? [y/N]: ', 'yellow', True)
            if input(notice).lower() == 'y':
                if client.clear_table():
                    ut.print_inf(f'Table "{ut.colorize(args.table_name, "red")}" has been cleared.')
                else:
                    ut.print_err(f'Failed to clear the table "{args.table_name}".')

        elif args.drop:
            if args.table_name is None:
                ut.print_err('Table name is required.')
                sys.exit(1)

            notice = ut.colorize(f'Are you sure to drop the table "{args.table_name}"? [y/N]: ', 'yellow', True)
            if input(notice).lower() == 'y':
                if client.drop_table():
                    ut.print_inf(f'Table "{ut.colorize(args.table_name, "red")}" has been dropped.')
                else:
                    ut.print_err(f'Failed to drop the table "{args.table_name}".')

    elif args.command == 'search':
        client = Client(table_name=args.table_name, bind=args.bind)
        # Validate similarity parameter
        if not 0.0 <= args.sim_thr <= 100.0:
            ut.print_err('Error: sim_thr must be between 0 and 100')
            sys.exit(1)

        ut.print_msg(f'Searching {args.target}...')
        results = client.search(args.target, args.num, args.sim_thr)
        if results:
            ut.print_inf(f'Found {len(results)} images similar to {Path(args.target).name}:', True)
            label_width = max(len(row['label']) for row in results)
            for row in results:
                print(f'  {row["label"]:<{label_width}}    {row["score"]}%')
                for path in row['paths']:
                    print(ut.colorize(f'  - {path}', 'gray', False))

            if args.open_res:
                paths = []
                for row in results:
                    if path := next((p for p in row['paths'] if Path(p).is_file()), None):
                        paths.append(path)
                    else:
                        ut.print_warn(f'No accessible path for image {row["id"]}')
                if paths:
                    ut.open_images(paths)
        elif results is None:
            ut.print_warn('Search queue is full, please try again later.')
            sys.exit(2)
        else:
            ut.print_warn(f'No image similar to {args.target} was found.')
            sys.exit(2)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
