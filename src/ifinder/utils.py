import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

RED = '\x1b[31m'
END = '\x1b[0m'
TRIM = '\x1b[K'
EXTENSIONS = Image.registered_extensions().keys()


def print_err(msg):
    """output error message"""
    print(f'{RED}{msg}{END}{TRIM}', file=sys.stderr)


def is_image(path: Path, ignore_hidden=True) -> bool:
    """check if path is an image"""
    if ignore_hidden and path.name[0] == '.':
        return False
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def find_all_images(paths: str | Path | Sequence[str | Path], recursively=True, ignore_hidden=True):
    """find all images in the given paths"""

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
