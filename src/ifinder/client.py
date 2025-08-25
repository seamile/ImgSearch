import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import Pyro5.api
import Pyro5.errors

from .storage import DB_DIR
from .utils import print_err

# The name for the service and the socket file.
SERVICE_NAME = 'ifinder.service'
SOCKET_NAME = 'ifinder.sock'


def create_parser() -> ArgumentParser:
    """Create command line argument parser"""
    parser = ArgumentParser(prog='ifinder', description='iFinder - A local image search engine.')

    # Main commands
    group = parser.add_mutually_exclusive_group()
    group.add_argument('query', nargs='?', help='Search query (image path or keyword)')
    group.add_argument('-a', dest='add', nargs='+', metavar='PATH', help='Add images to DB (file or directory path)')
    group.add_argument('-i', dest='info', action='store_true', help='Show database information')
    group.add_argument('-C', dest='clear', action='store_true', help='Clear the entire database')
    group.add_argument('-c', dest='compare', nargs=2, metavar='IMG_PATH', help='Compare similarity of two images')

    # Service management commands
    srv_cmds = ['start', 'stop', 'restart', 'status']
    group.add_argument('-s', dest='service', choices=srv_cmds, help='Manage the iFinder service')

    # Optional arguments
    parser.add_argument(
        '-d', dest='database', type=Path, default=DB_DIR, help=f'Database directory path (default: {DB_DIR})'
    )
    parser.add_argument('-n', dest='num', type=int, default=10, help='Number of search results (default: 10)')
    parser.add_argument('-m', dest='model', type=str, help='CLIP model name for the service to use')
    parser.add_argument(
        '-D', dest='daemon', action='store_true', help='Run the service in the background (used with -s start/restart)'
    )

    return parser


def _handle_service_command(args: Namespace):
    """
    Handles service management by locally importing the necessary (heavy) modules.
    This keeps the client lightweight for non-service commands.
    """
    # This import is intentionally local to avoid loading heavy modules for regular commands.
    from .server import ServerDaemon

    daemon = ServerDaemon()
    if args.service == 'start':
        daemon.start(daemonize=args.daemon)
    elif args.service == 'stop':
        daemon.stop()
    elif args.service == 'restart':
        daemon.restart(daemonize=args.daemon)
    elif args.service == 'status':
        daemon.status()


def _connect_to_service() -> Pyro5.api.Proxy:
    """Connect to the Pyro5 service via UDS and return the proxy object."""
    uds_socket = DB_DIR / SOCKET_NAME
    if not uds_socket.exists():
        print_err(f"Service not running or socket file missing at '{uds_socket}'.")
        print_err('You can start the service with: ifinder -s start')
        sys.exit(1)

    try:
        uri = f'PYRO:{SERVICE_NAME}@./u:{uds_socket}'
        service = Pyro5.api.Proxy(uri)
        service._pyroBind()  # A quick check to see if the server is responsive
        return service
    except Pyro5.errors.CommunicationError:
        print_err(f"Failed to connect to service socket at '{uds_socket}'.")
        print_err('Is the iFinder service running? Check with: ifinder -s status')
        sys.exit(1)
    except Exception as e:
        print_err(f'An unexpected error occurred while connecting to the service: {e}')
        sys.exit(1)


def _handle_add(service: Pyro5.api.Proxy, args: Namespace):
    print('Sending request to add images...')
    added_count: int = service.add_images(args.add)  # type: ignore
    if added_count > 0:
        print(f'Service successfully added {added_count} images to the index.')
    else:
        print('Service reported that no new images were added.')


def _handle_search(service: Pyro5.api.Proxy, args: Namespace):
    print('Searching...')
    results: list = service.search(args.query, k=args.num)  # type: ignore
    if results:
        print(f'\nFound {len(results)} similar results:')
        print('-' * 60)
        for i, (path, similarity) in enumerate(results, 1):
            print(f'{i:2d}. {path} (similarity: {similarity}%)')
    else:
        print('No similar images found.')


def _handle_info(service: Pyro5.api.Proxy, args: Namespace):
    info: dict = service.get_db_info()  # type: ignore
    print('Database Information:')
    for key, value in info.items():
        print(f'  - {key.replace("_", " ").title()}: {value}')


def _handle_clear(service: Pyro5.api.Proxy, args: Namespace):
    if input('Are you sure you want to clear the entire database? [y/N]: ').lower() == 'y':
        if service.clear_db():
            print('Database has been cleared.')
        else:
            print_err('Failed to clear the database.')
    else:
        print('Operation cancelled.')


def _handle_compare(service: Pyro5.api.Proxy, args: Namespace):
    path1, path2 = args.compare
    similarity = service.compare_images(path1, path2)
    print(f'Similarity between images: {similarity}%')


def _execute_command(service: Pyro5.api.Proxy, args: Namespace, parser: ArgumentParser):
    """Execute the command based on parsed arguments."""
    if args.add:
        _handle_add(service, args)
    elif args.query:
        _handle_search(service, args)
    elif args.info:
        _handle_info(service, args)
    elif args.clear:
        _handle_clear(service, args)
    elif args.compare:
        _handle_compare(service, args)
    else:
        parser.print_help()


def main() -> None:
    """Main function for command line interface"""
    parser = create_parser()
    args = parser.parse_args()

    # Delegate service management commands
    if args.service:
        _handle_service_command(args)
        sys.exit(0)

    # Connect to the service and execute the command
    service = _connect_to_service()
    try:
        _execute_command(service, args, parser)
    except Pyro5.errors.CommunicationError as e:
        print_err(f'Communication error: {e}')
        print_err('The service might have stopped. Please check its status.')
        sys.exit(1)
    except Exception as e:
        print_err(f'An unexpected error occurred: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
