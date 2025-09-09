import logging
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from io import BytesIO
from itertools import islice
from pathlib import Path

from PIL import Image

from imgsearch.consts import BASE_DIR

EXTENSIONS = Image.registered_extensions().keys()

Feature = list[float]


def colorize(text: str, color: str = '', bold=False) -> str:
    """Colorize text"""
    b = '1' if bold else '0'
    match color.lower():
        case 'red':
            return f'\x1b[{b};31m{text}\x1b[0m'
        case 'green':
            return f'\x1b[{b};32m{text}\x1b[0m'
        case 'yellow':
            return f'\x1b[{b};33m{text}\x1b[0m'
        case 'blue':
            return f'\x1b[{b};34m{text}\x1b[0m'
        case 'magenta':
            return f'\x1b[{b};35m{text}\x1b[0m'
        case 'cyan':
            return f'\x1b[{b};36m{text}\x1b[0m'
        case 'white':
            return f'\x1b[{b};37m{text}\x1b[0m'
        case 'gray':
            return f'\x1b[{b};90m{text}\x1b[0m'
        case _:
            return f'\x1b[{b}m{text}\x1b[0m'


def bold(text: str) -> str:
    """Bold text"""
    return colorize(text, '', True)


def print_warn(msg):
    """Output warning message to stderr"""
    print(colorize(msg, 'yellow'), file=sys.stderr)


def print_err(msg):
    """Output error message to stderr"""
    print(colorize(msg, 'red'), file=sys.stderr)


def cpu_count() -> int:
    """Return the number of CPUs"""
    return os.cpu_count() or 1


def is_image(path: Path, ignore_hidden=True) -> bool:
    """Check if the given path is an image file"""
    if ignore_hidden and path.name[0] == '.':
        return False
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def img2bytes(img: Image.Image, resize: int = 0) -> bytes:
    """Convert image to bytes"""
    if resize > 0:
        img.thumbnail((resize, resize))
    buffer = BytesIO()
    img.convert('RGB').save(buffer, format='webp', quality=97)
    return buffer.getvalue()


def bytes2img(img_bytes: bytes) -> Image.Image:
    """Convert bytes to image"""
    return Image.open(BytesIO(img_bytes))


def find_all_images(paths: str | Path | Sequence[str | Path], recursively=True, ignore_hidden=True):
    """Find all image files in the given paths"""

    if isinstance(paths, (str, Path)):
        paths = [paths]

    for path in paths:
        path = path if isinstance(path, Path) else Path(path)

        if path.is_dir():
            targets = path.rglob('*') if recursively else path.iterdir()
            for fpath in targets:
                if is_image(fpath, ignore_hidden):
                    yield fpath

        elif path.is_file():
            if is_image(path, ignore_hidden):
                yield path

        else:
            print_err(f'{path} is not a file or directory')


def ibatch(iterable, batch_size):
    """Batch iterator"""
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


class ColorFormatter(logging.Formatter):
    """Color formatter for logging"""

    @staticmethod
    def colorize(level: str, message: str) -> str:
        colors = {
            'DEBUG': 'gray',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
        return colorize(message, colors.get(level.upper(), ''))

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return self.colorize(record.levelname, message)


def get_logger(name: str, level: int = logging.INFO, log_dir=BASE_DIR):
    """Setup logging based on TTY status"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers to avoid duplication

    # Check if running in TTY (foreground) or background
    handler: logging.Handler
    if sys.stderr.isatty():
        # Foreground: use console handler with colors
        handler = logging.StreamHandler()
        handler.setLevel(level)
    else:
        # Background: use file handler
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'isearch.log'
        handler = logging.FileHandler(log_file)
        handler.setLevel(level)
    formatter = ColorFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger


def open_images(paths: Sequence[str | Path]):
    """Open images with system default image viewer"""
    system = platform.system()
    try:
        if system == 'Windows':
            subprocess.run(['explorer', *paths])
        elif system == 'Darwin':
            subprocess.run(['open', *paths])
        else:
            subprocess.run(['xdg-open', *paths])
    except Exception as e:
        print_err(f'Failed to open images: {e}')
