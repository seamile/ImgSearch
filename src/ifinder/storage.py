from pathlib import Path
from pickle import dump, load  # noqa: S403

from hnswlib import Index

DEFAULT_INDEX = Path.home() / '.ifinder' / '.ifinder.db'


class ImgBase:
    """Image feature index"""

    def __init__(self) -> None:
        self.idx_mapping: dict[int, str] = {}

    def new_index(self, path: Path):
        """create a new index"""
        self.index = Index(space='cosine', dim=512)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16, allow_replace_delete=True)

    def next_max_elements(self):
        """get next max elements"""
        return round(self.index.max_elements * 1.6)

    def add_image(self, path: Path, feature: list[float]):
        """add one image to index"""
        pass

    def add_images(self, paths: list[Path], features: list[list[float]]):
        """add images to index"""
        pass

    def search(self, feature: list[float], k: int = 10):
        """search images by feature vector"""
        pass

    @classmethod
    def load_db(cls, path: Path):
        load(path.open('rb'))  # noqa: S301

    def save(self, path: Path):
        dump(self.index, path.open('wb'))
