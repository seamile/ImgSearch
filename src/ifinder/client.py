import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import Pyro5.api
import Pyro5.errors

from .storage import DB_DIR
from .utils import print_err

# The name for the service and the socket file.
SERVICE_NAME = 'ifinder.service'
SOCKET_NAME = 'ifinder.sock'


class Client:
    """
    iFinder client for interacting with the iFinder service.

    This client provides a command-line interface to the iFinder service,
    supporting image search, database management, and service control operations.
    """

    def __init__(self) -> None:
        """Initialize the iFinder client."""
        self.service: Any  # The Pyro5 proxy to the iFinder service
        self.args: Namespace

    def create_parser(self) -> ArgumentParser:
        """Create command line argument parser."""
        parser = ArgumentParser(prog='ifinder', description='iFinder - A local image search engine.')

        # Main commands
        group = parser.add_mutually_exclusive_group()
        group.add_argument('query', nargs='?', help='Search query (image path or keyword)')
        group.add_argument(
            '-a', dest='add', nargs='+', metavar='PATH', help='Add images to DB (file or directory path)'
        )
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
            '-l',
            dest='label',
            choices=['path', 'name'],
            default='path',
            help='Label naming method: "path" for absolute path, "name" for filename without extension (default: path)',
        )
        parser.add_argument(
            '-D',
            dest='daemon',
            action='store_true',
            help='Run the service in the background (used with -s start/restart)',
        )

        return parser

    def run(self):
        """Run the client with command line arguments."""
        parser = self.create_parser()
        self.args = parser.parse_args()

        # Delegate service management commands
        if self.args.service:
            self._handle_service_command()
            return

        # Connect to service for other operations
        self.service = self._connect_to_service()

        # Handle different operations
        if self.args.add:
            self._handle_add()
        elif self.args.info:
            self._handle_info()
        elif self.args.clear:
            self._handle_clear()
        elif self.args.compare:
            self._handle_compare()
        elif self.args.query:
            self._handle_search()
        else:
            parser.print_help()

    def _handle_service_command(self):
        """Handle service management commands."""
        # This import is intentionally local to avoid loading heavy modules for regular commands.
        from .server import ServerDaemon

        daemon = ServerDaemon()
        if self.args.service == 'start':
            daemon.start(daemonize=self.args.daemon)
        elif self.args.service == 'stop':
            daemon.stop()
        elif self.args.service == 'restart':
            daemon.restart(daemonize=self.args.daemon)
        elif self.args.service == 'status':
            daemon.status()

    def _connect_to_service(self) -> Pyro5.api.Proxy:
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

    def _handle_add(self):
        """Handle adding images to the index."""
        print('Sending request to add images...')
        # Convert paths to absolute paths
        abs_paths = [str(Path(p).resolve()) for p in self.args.add]
        added_count: int = self.service.add_images(abs_paths, label_type=self.args.label)  # type: ignore
        if added_count > 0:
            print(f'Service successfully added {added_count} images to the index.')
        else:
            print('Service reported that no new images were added.')

    def _handle_search(self):
        """Handle search operations."""
        print('Searching...')
        query = self.args.query
        # If query is a file path, convert to absolute path
        if Path(query).exists():
            query = str(Path(query).resolve())
        results: list = self.service.search(query, k=self.args.num)  # type: ignore
        if results:
            print(f'\nFound {len(results)} similar results:')
            print('-' * 60)
            for i, (path, similarity) in enumerate(results, 1):
                print(f'{i:2d}. {path} (similarity: {similarity}%)')
        else:
            print('No similar images found.')

    def _handle_info(self):
        """Handle database info request."""
        info: dict = self.service.get_db_info()  # type: ignore
        print('Database Information:')
        for key, value in info.items():
            print(f'  - {key.replace("_", " ").title()}: {value}')

    def _handle_clear(self):
        """Handle database clear request."""
        if __name__ != '__main__':
            return self.service.clear_db()
        elif input('Are you sure you want to clear the entire database? [y/N]: ').lower() == 'y':
            if self.service.clear_db():
                print('Database has been cleared.')
            else:
                print_err('Failed to clear the database.')
        else:
            print('Operation cancelled.')

    def _handle_compare(self):
        """Handle image comparison request."""
        path1, path2 = self.args.compare
        # Convert to absolute paths
        abs_path1 = str(Path(path1).resolve())
        abs_path2 = str(Path(path2).resolve())
        similarity = self.service.compare_images(abs_path1, abs_path2)
        print(f'Similarity between images: {similarity}%')


def main() -> None:
    """Main function for command line interface"""
    client = Client()
    client.run()


if __name__ == '__main__':
    main()
