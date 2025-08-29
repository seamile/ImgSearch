import logging
import platform
import subprocess  # noqa: S404
import sys
from collections.abc import Sequence
from io import BytesIO
from itertools import islice
from pathlib import Path

from PIL import Image

from pixa.consts import BASE_DIR

# 定义颜色
RED = '\x1b[31m'  # 红色
GREEN = '\x1b[32m'  # 绿色
YELLOW = '\x1b[33m'  # 黄色
BLUE = '\x1b[34m'  # 蓝色
MAGENTA = '\x1b[35m'  # 品红
CYAN = '\x1b[36m'  # 青色
WHITE = '\x1b[37m'  # 白色
GRAY = '\x1b[90m'  # 灰色
END = '\x1b[0m'

EXTENSIONS = Image.registered_extensions().keys()

Feature = list[float]


def print_warn(msg):
    """Output warning message to stderr"""
    print(f'{YELLOW}{msg}{END}', file=sys.stderr)


def print_err(msg):
    """Output error message to stderr"""
    print(f'{RED}{msg}{END}', file=sys.stderr)


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
        match level:
            case 'DEBUG':
                return f'{GRAY}{message}{END}'
            case 'INFO':
                return f'{BLUE}{message}{END}'
            case 'WARNING':
                return f'{YELLOW}{message}{END}'
            case 'ERROR':
                return f'{RED}{message}{END}'
            case 'CRITICAL':
                return f'{RED}{message}{END}'
            case _:
                return message

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
        log_file = log_dir / 'pixa.log'
        handler = logging.FileHandler(log_file)
        handler.setLevel(level)
    formatter = ColorFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger


def open_images(paths: list[str | Path]):
    """Open images with system default image viewer"""
    system = platform.system()
    try:
        if system == 'Windows':
            subprocess.run(['explorer', *paths])  # type: ignore  # noqa: S603, S607
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', *paths])  # type: ignore  # noqa: S603, S607
        else:  # Linux
            subprocess.run(['xdg-open', *paths])  # type: ignore  # noqa: S603, S607
    except Exception as e:
        print_err(f'Failed to open images: {e}')
