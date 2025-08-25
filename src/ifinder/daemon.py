import os
import sys
import time
from pathlib import Path
from signal import SIGTERM

from .storage import DB_DIR
from .utils import print_err

PID_FILE = DB_DIR / 'ifinder.pid'
LOG_FILE = DB_DIR / 'ifinder.log'


class Daemon:
    """A generic daemon class."""

    def __init__(self, pidfile: Path = PID_FILE, stdout: Path = LOG_FILE, stderr: Path = LOG_FILE):
        self.stdout = stdout
        self.stderr = stderr
        self.pidfile = pidfile

    def _daemonize(self):
        """Deamonize process using the classic double-fork mechanism."""
        try:
            if os.fork() > 0:
                sys.exit(0)
        except OSError as e:
            print_err(f'fork #1 failed: {e}')
            sys.exit(1)

        os.chdir('/')
        os.setsid()
        os.umask(0)

        try:
            if os.fork() > 0:
                sys.exit(0)
        except OSError as e:
            print_err(f'fork #2 failed: {e}')
            sys.exit(1)

        sys.stdout.flush()
        sys.stderr.flush()
        with open(self.stdout, 'ab', 0) as so, open(self.stderr, 'ab', 0) as se:
            os.dup2(so.fileno(), sys.stdout.fileno())
            os.dup2(se.fileno(), sys.stderr.fileno())

        pid = str(os.getpid())
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        self.pidfile.write_text(pid + '\n')

    def get_pid(self) -> int | None:
        try:
            return int(self.pidfile.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    def is_running(self) -> bool:
        pid = self.get_pid()
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    def start(self, daemonize: bool = True):
        """Start the service."""
        if self.is_running():
            print_err('Service is already running.')
            sys.exit(1)

        if daemonize:
            print('Starting service in daemon mode...')
            self._daemonize()
            self.run()
        else:
            print('Starting service in foreground mode... (Press Ctrl+C to stop)')
            pid = str(os.getpid())
            self.pidfile.parent.mkdir(parents=True, exist_ok=True)
            self.pidfile.write_text(pid + '\n')
            try:
                self.run()
            finally:
                if self.pidfile.exists() and self.get_pid() == os.getpid():
                    self.pidfile.unlink()

    def stop(self):
        """Stop the daemon."""
        pid = self.get_pid()
        if not pid:
            print_err("pidfile not found. Is the service running?")
            return

        print('Stopping service...')
        try:
            while True:
                os.kill(pid, SIGTERM)
                time.sleep(0.1)
        except OSError:
            pass

        if self.pidfile.exists():
            self.pidfile.unlink()

    def restart(self, daemonize: bool = True):
        """Restart the daemon."""
        self.stop()
        time.sleep(0.5)  # Give it a moment to release resources
        self.start(daemonize=daemonize)

    def status(self):
        """Get the status of the daemon."""
        if self.is_running():
            pid = self.get_pid()
            print(f'Service is running with pid {pid}.')
        else:
            print('Service is not running.')

    def run(self):
        """You should override this method when you subclass Daemon."""
        raise NotImplementedError