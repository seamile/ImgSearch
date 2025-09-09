from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump, load

from bidict import bidict
from hnswlib import Index

from imgsearch.consts import BASE_DIR, CAPACITY, DB_NAME, IDX_NAME, MAP_NAME
from imgsearch.utils import Feature

Mapping = bidict[int, str]


class VectorDB:
    """Vector database for storing and searching item features"""

    def __init__(self, db_name: str = DB_NAME, base_dir: Path = BASE_DIR) -> None:
        self.name = db_name
        self.base = base_dir
        self.path = (base_dir / db_name).resolve()
        self.idx_path = self.path / IDX_NAME
        self.map_path = self.path / MAP_NAME
        self.index, self.mapping = self.load_db(self.path)

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

    def has_id(self, id: int) -> bool:  # noqa: A002
        """Check if id exists in index"""
        return id in self.index.get_ids_list()

    def has_label(self, label: str) -> bool:
        """Check if label exists in index"""
        return label in self.mapping.inv

    def has_labels(self, *labels: str) -> list[bool]:
        """Check if labels exists in index"""
        return [label in self.mapping.inv for label in labels]

    @staticmethod
    def new_index(init=True) -> Index:
        index = Index(space='cosine', dim=512)
        if init is True:
            index.init_index(max_elements=CAPACITY, ef_construction=200, M=16, allow_replace_deleted=True)  # type: ignore
        return index

    @classmethod
    def load_db(cls, db_path: Path | str) -> tuple[Index, Mapping]:
        """Load database from file, or create a new one if not exist"""
        if isinstance(db_path, str):
            db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        idx_path = db_path / IDX_NAME
        map_path = db_path / MAP_NAME
        if not idx_path.exists() and not map_path.exists():
            index = cls.new_index(init=True)
            mapping: Mapping = bidict()
        elif idx_path.is_file() and map_path.is_file():
            # load index file
            index = cls.new_index(init=False)
            index.load_index(str(idx_path), allow_replace_deleted=True)  # type: ignore

            # load mapping file
            with map_path.open('rb') as fp:
                mapping = load(fp)  # noqa: S301
            if not isinstance(mapping, bidict):
                mapping = bidict(mapping)

            # check if the index and mapping files are consistent
            if index.element_count != len(mapping):
                raise ValueError('Index and mapping files are not consistent')
        else:
            raise OSError('DB file may be corrupted')

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

    def add_items(self, labels: list[str], features: list[Feature], override: bool = False):
        """Add multiple items to index"""
        # TODO:
        # Check whether the label already exists. If override is True,
        # overwrite the old values, otherwise ignore existing labels

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

    def save(self):
        """Save database to file"""
        # Save index
        self.index.save_index(str(self.idx_path))

        # Save mapping
        with self.map_path.open('wb') as f:
            dump(self.mapping, f, protocol=HIGHEST_PROTOCOL)

    def clear(self):
        """Clear database"""
        self.index = self.new_index()
        self.mapping = bidict()
        self.save()

    def search(self, feature: Feature, k: int = 10, similarity: float = 0.0) -> list[tuple[str, float]]:
        """Search items by feature vector with similarity filtering"""
        if self.index is None or self.size == 0:
            return []

        # Validate similarity parameter
        if similarity < 0.0 or similarity > 100.0:
            raise ValueError('similarity must be between 0 and 100')

        # If no similarity filtering, use original behavior
        if similarity == 0.0:
            v_ids, distances = self.index.knn_query([feature], k=min(k, self.size))
        else:
            # Dynamic search with similarity filtering
            search_k = min(k * 3, self.size)
            v_ids, distances = self.index.knn_query([feature], k=search_k)

        # Convert results to (path, similarity) tuples with filtering
        results = []
        for vid, distance in zip(v_ids[0], distances[0], strict=True):
            if vid in self.mapping:
                # Convert distance to similarity
                res_similarity = round((1.0 - distance) * 100)
                if res_similarity >= similarity:
                    results.append((self.mapping[vid], res_similarity))
                    # break if results are enough
                    if len(results) >= k:
                        break

        return results[:k]
