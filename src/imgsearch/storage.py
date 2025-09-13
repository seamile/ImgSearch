"""Vector Database Module for ImgSearch

This module implements a lightweight vector database using HNSWLIB for approximate
nearest neighbor search and bidict for label-ID mapping. Supports persistence via
binary index files and pickled mappings. Designed for CLIP embeddings (512-dim,
cosine similarity).

Architecture:
- HNSW Index: Hierarchical Navigable Small World graph for fast ANN search.
  - Space: 'cosine' for normalized vectors.
  - Params: ef_construction=400 (build quality), M=32 (connections), allow_replace_deleted=True.
- Bidict Mapping: Maintains bidirectional ID<->label lookup for O(1) access.
- Persistence: index.db (HNSW binary), mapping.db (pickled dict).
- Auto-resize: Increases capacity by 10k when full (initial CAPACITY=10k).

Usage:
    db = VectorDB('my_db')
    db.add_items(labels, features)  # Add CLIP vectors
    results = db.search(query_feature, k=10, similarity=80)  # Top-k with threshold
    db.save()  # Persist changes

Limitations:
- Single-threaded writes (no distributed locking).
- Pickle serialization (security risk for untrusted data).
- Fixed dim=512 (CLIP-specific).
"""

from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump, load

from bidict import bidict
from hnswlib import Index

from imgsearch.consts import BASE_DIR, CAPACITY, DB_NAME, IDX_NAME, MAP_NAME
from imgsearch.utils import Feature, ibatch

# Type alias for ID-label mapping
Mapping = bidict[int, str]


class VectorDB:
    """Vector database using HNSW for ANN search and bidict for label mapping.

    Stores 512-dim CLIP features with cosine similarity. Supports batch add/search,
    auto-resizing index, persistence, and duplicate checking. Thread-safe for reads;
    external locking needed for concurrent writes.

    Lifecycle:
    - Init: Loads or creates index/mapping from files.
    - Add: Embeds labels/features, resizes if needed, updates mapping.
    - Search: Dynamic ef adjustment based on k for balance speed/accuracy.
    - Save: Writes index binary and pickled mapping.
    - Clear: Resets to empty state.

    Example:
        db = VectorDB('search_db')
        db.add_items(['img1', 'img2'], [[0.1]*512, [0.2]*512])
        matches = db.search([0.15]*512, k=5, similarity=70)
    """

    def __init__(self, db_name: str = DB_NAME, base_dir: Path = BASE_DIR) -> None:
        """Initialize or load VectorDB instance.

        Creates paths, loads existing DB if files present, or initializes empty.
        Validates consistency between index count and mapping size.

        Args:
            db_name (str): Database identifier. Defaults to DB_NAME ('default').
            base_dir (Path): Root for DB directories. Defaults to BASE_DIR (~/.isearch).
        """
        self.name = db_name
        self.base = base_dir
        self.path = (base_dir / db_name).resolve()
        self.idx_path = self.path / IDX_NAME  # HNSW index file
        self.map_path = self.path / MAP_NAME  # Label mapping file
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
    def new_index(init=True, max_elements: int = CAPACITY, ef: int = 400, max_conn: int = 32) -> Index:
        index = Index(space='cosine', dim=512)
        if init is True:
            index.init_index(max_elements=max_elements, ef_construction=ef, M=max_conn, allow_replace_deleted=True)  # type: ignore
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

    def db_list(self) -> list[str]:
        """List all available database names in base directory."""
        if not self.base.exists():
            return []

        databases: list[str] = [
            item.name
            for item in self.base.iterdir()
            if item.is_dir() and (item / IDX_NAME).is_file() and (item / MAP_NAME).is_file()
        ]

        return sorted(databases)

    def rebuild_index(self, ef: int, max_conn: int):
        """Rebuild index with new ef parameter"""
        old_index = self.index
        new_index = self.new_index(init=True, max_elements=self.capacity, ef=ef, max_conn=max_conn)
        ids = old_index.get_ids_list()
        if len(ids) > 0:
            for batch_ids in ibatch(ids, batch_size=1000):
                vectors_batch = old_index.get_items(batch_ids)
                new_index.add_items(vectors_batch, batch_ids, replace_deleted=True)
        self.index = new_index
        self.save()

    def search(self, feature: Feature, k: int = 10, similarity: float = 0.0) -> list[tuple[str, float]]:
        """Search items by feature vector with similarity filtering"""
        if self.index is None or self.size == 0 or not feature:
            return []

        # Validate similarity parameter
        if similarity < 0.0 or similarity > 100.0:
            raise ValueError('similarity must be between 0 and 100')

        # Set ef to a value between 150 and 300, depending on the number of results requested.
        # This is to ensure that the search is efficient and fast, without sacrificing accuracy.
        search_k = min(k, self.size)
        self.index.set_ef(min(max(search_k * 3, 150), 300))
        v_ids, distances = self.index.knn_query([feature], k=search_k)

        # Convert results to (label, similarity) tuples
        results = []
        for vid, distance in zip(v_ids[0], distances[0], strict=True):
            if label := self.mapping.get(vid):
                # Convert distance to similarity
                res_similarity = round((1.0 - distance) * 100, 1)
                if res_similarity >= similarity:
                    results.append((label, res_similarity))

        return results
