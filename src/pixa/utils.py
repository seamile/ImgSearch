import logging
import sys
from collections.abc import Sequence
from io import BytesIO
from itertools import islice
from pathlib import Path

from PIL import Image

# 定义颜色
RED = '\x1b[31m'  # 红色
YELLOW = '\x1b[33m'  # 黄色
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


def get_logger(name: str, level: int = logging.INFO):
    """Setup logging"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    color_formatter = ColorFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(color_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger
