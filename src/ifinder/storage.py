from pathlib import Path
from pickle import dump, load  # noqa: S403

from hnswlib import Index

from ifinder.utils import Feature

DB_DIR = Path.home() / '.ifinder'
IDX_NAME = 'index.bin'
MAP_NAME = 'mapping.db'
CAPACITY = 10000


class VectorDB:
    """Vector database for storing and searching item features"""

    def __init__(self, db_dir: Path = DB_DIR) -> None:
        self.base_dir = db_dir
        try:
            self.index, self.mapping = self.load_db(self.base_dir)
        except Exception:
            self.index = self.new_index()
            self.mapping = {}

    @property
    def size(self) -> int:
        return self.index.element_count

    @property
    def next_id(self) -> int:
        return self.index.element_count + 1

    @property
    def capacity(self) -> int:
        return self.index.max_elements

    @property
    def next_capacity(self) -> int:
        """Get next max elements for resizing index"""
        return self.index.max_elements + CAPACITY

    @staticmethod
    def new_index(init=True) -> Index:
        index = Index(space='cosine', dim=512)
        if init is True:
            index.init_index(max_elements=CAPACITY, ef_construction=200, M=16, allow_replace_deleted=True)  # type: ignore
        return index

    @classmethod
    def load_db(cls, base_dir: Path | str) -> tuple[Index, dict[int, str]]:
        """Load database from file"""
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)

        idx_path = base_dir / IDX_NAME
        map_path = base_dir / MAP_NAME
        if not idx_path.is_file() or not map_path.is_file():
            raise FileNotFoundError(f'Database file not found: {idx_path}, {map_path}')

        # load index file
        index = cls.new_index(init=False)
        index.load_index(idx_path.as_posix(), allow_replace_deleted=True)  # type: ignore

        # load mapping file
        with map_path.open('rb') as fp:
            mapping = load(fp)  # noqa: S301

        return index, mapping

    def add_item(self, label: str, feature: Feature):
        """Add one item to index"""
        # Check if we need to resize the index
        if self.size >= self.capacity:
            self.index.resize_index(self.next_capacity)

        # Add the feature vector to the index
        item_id = self.next_id
        self.index.add_items([feature], [item_id], replace_deleted=True)
        self.mapping[item_id] = label

    def add_items(self, labels: list[str], features: list[Feature]):
        """Add multiple items to index"""
        incr_size = len(features)
        if incr_size <= 0 or len(labels) != incr_size:
            raise ValueError('Invalid labels or features')

        # Check if we need to resize the index
        if self.size + incr_size > self.capacity:
            self.index.resize_index(self.next_capacity)

        # Prepare IDs and update mapping
        ids = list(range(self.next_id, self.next_id + incr_size))
        for i, label in enumerate(labels):
            self.mapping[self.next_id + i] = label

        # Add features to index
        self.index.add_items(features, ids, replace_deleted=True)

    def save(self, base_dir: Path | None = None):
        """Save database to file"""
        if base_dir is None:
            base_dir = self.base_dir

        # Ensure parent directory exists
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save index
        idx_path = base_dir / IDX_NAME
        self.index.save_index(idx_path.as_posix())

        # Save mapping
        map_path = base_dir / MAP_NAME
        with map_path.open('wb') as f:
            dump(self.mapping, f)

    def clear(self):
        """Clear database"""
        self.index = self.new_index()
        self.mapping.clear()
        self.save()

    def search(self, feature: Feature, k: int = 10) -> list[tuple[str, float]]:
        """Search items by feature vector"""
        if self.index is None or self.size == 0:
            return []

        # Search for similar items
        v_ids, distances = self.index.knn_query([feature], k=min(k, self.size))

        # Convert results to (path, similarity) tuples
        results = []
        for vid, distance in zip(v_ids[0], distances[0], strict=True):
            if vid in self.mapping:
                # Convert distance to similarity
                similarity = round((1.0 - distance) * 100)
                results.append((self.mapping[vid], similarity))

        return results
