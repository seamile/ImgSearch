"""Vector Database Module for ImgSearch

This module implements a lightweight vector database using HNSWLIB for approximate
nearest neighbor search and bidict for label-ID mapping. Supports persistence via
binary index files and pickled mappings.

Architecture:
- HNSW Index: Hierarchical Navigable Small World graph for fast ANN search.
- Bidict Mapping: Maintains bidirectional ID<->label lookup for O(1) access.
- Persistence: index.db (HNSW binary), mapping.db (pickled dict).
- Auto-resize: Increases capacity by 10k when full (initial cfg.CAPACITY=10k).

Limitations:
- Single-threaded writes (no distributed locking).
- Pickle serialization (security risk for untrusted data).
"""

import shutil
from collections.abc import Iterable
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump, load
from threading import RLock

from bidict import bidict
from hnswlib import Index

from imgsearch import config as cfg
from imgsearch.utils import Feature, ibatch, multi_pop, multi_remove

# Type alias for ID-label mapping
Mapping = bidict[int, str]


class VectorDB:
    """Vector database using HNSW for ANN search and bidict for label mapping.

    Stores CLIP features with cosine similarity. Supports batch add, search,
    auto-resizing index, persistence, and duplicate checking. Thread-safe for reads;
    external locking needed for concurrent writes.

    Example:
        db = VectorDB('search_db', dim=512)
        db.add_items(['img1', 'img2'], [[0.1]*512, [0.2]*512])
        matches = db.search([0.15]*512, k=5, similarity=70)
    """

    def __init__(self, db_name: str = cfg.DB_NAME, base_dir: Path = cfg.BASE_DIR, dim: int = 512) -> None:
        """Initialize or load VectorDB instance.

        Creates paths, loads existing DB if files present, or initializes empty.
        Validates consistency between index count and mapping size.

        Args:
            db_name (str): Database identifier. Defaults to cfg.DB_NAME ('default').
            base_dir (Path): Root for DB directories. Defaults to cfg.BASE_DIR (~/.isearch).
        """
        self.name = db_name
        self.base = base_dir
        self.dim = dim
        self.path = (base_dir / db_name).resolve()
        self.idx_path = self.path / cfg.IDX_NAME  # HNSW index file
        self.map_path = self.path / cfg.MAP_NAME  # Label mapping file
        self.index, self.mapping = self.load_db(self.path, dim)
        self.wlock = RLock()

    def __len__(self) -> int:
        """Get number of items in mapping"""
        return len(self.mapping)

    def __contains__(self, key: int | str) -> bool:
        """Check if id or label exists in index"""
        if isinstance(key, int):
            return key in self.mapping
        else:
            return key in self.mapping.inv

    def __getitem__(self, key: int | str) -> Feature:
        """Get feature vector for id or label"""
        try:
            if isinstance(key, int):
                features = self.index.get_items([key])
            else:
                fid = self.mapping.inv[key]
                features = self.index.get_items([fid])
        except (RuntimeError, KeyError) as e:
            raise KeyError(f'Feature id or label "{key}" not found') from e

        if len(features) != 1:
            raise KeyError(f'Feature id or label "{key}" not found')
        return features[0].tolist()  # type: ignore

    @property
    def size(self) -> int:
        """Get disk usage of index and mapping file"""
        return self.idx_path.stat().st_size + self.map_path.stat().st_size

    @property
    def count(self) -> int:
        """Get number of items in mapping"""
        return len(self)

    @property
    def next_id(self) -> int:
        """Get next id for adding item"""
        return self.index.element_count + 1

    @property
    def capacity(self) -> int:
        """Get current max elements for index"""
        return self.index.max_elements

    @property
    def next_capacity(self) -> int:
        """Get next max elements for resizing index"""
        return self.index.max_elements + cfg.CAPACITY

    @staticmethod
    def new_index(
        init=True,
        dim: int = 512,
        max_elements: int = cfg.CAPACITY,
        ef: int = 400,
        max_conn: int = 32,
    ) -> Index:
        """Create new HNSW index"""
        index = Index(space='cosine', dim=dim)
        if init is True:
            index.init_index(
                max_elements=max_elements,
                ef_construction=ef,
                M=max_conn,
                allow_replace_deleted=True,  # type: ignore
            )
        return index

    @classmethod
    def load_db(cls, db_path: Path | str, dim: int = 512) -> tuple[Index, Mapping]:
        """Load database from file, or create a new one if not exist"""
        if isinstance(db_path, str):
            db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        idx_path = db_path / cfg.IDX_NAME
        map_path = db_path / cfg.MAP_NAME
        if not idx_path.exists() and not map_path.exists():
            index = cls.new_index(init=True, dim=dim)
            mapping: Mapping = bidict()
        elif idx_path.is_file() and map_path.is_file():
            # load index file
            index = cls.new_index(init=False, dim=dim)
            index.load_index(str(idx_path), allow_replace_deleted=True)  # type: ignore

            # load mapping file
            with map_path.open('rb') as fp:
                mapping = load(fp)  # noqa: S301
            if not isinstance(mapping, bidict):
                mapping = bidict(mapping)

            # check if the index and mapping files are consistent
            feature_ids = mapping.keys()
            if feature_ids != feature_ids & index.get_ids_list():
                raise ValueError('Index and mapping files are not consistent')
        else:
            raise OSError('DB file may be corrupted')

        return index, mapping

    def add_item(self, label: str, feature: Feature, overwrite: bool = True) -> bool:
        """Add one item to index"""
        with self.wlock:
            if label in self.mapping.inv:
                if not overwrite:
                    return False
                fid = self.mapping.inv[label]
            else:
                # Check if we need to resize the index
                if self.count >= self.capacity:
                    self.index.resize_index(self.next_capacity)

                # Add the feature vector to the index
                fid = self.next_id
                self.mapping[fid] = label
            self.index.add_items([feature], [fid], replace_deleted=True)
            self.save()
            return True

    def add_items(self, labels: list[str], features: list[Feature], *, overwrite: bool = True) -> int:
        """Add multiple items to index"""
        if len(labels) != len(features):
            raise ValueError('Labels and features must be of the same length')

        updated = 0
        with self.wlock:
            # Filter out existing labels
            if existing_labels := sorted(self.mapping.inv.keys() & set(labels)):
                indices = multi_remove(labels, existing_labels)
                features_to_overwrite = multi_pop(features, indices)
                if len(existing_labels) != len(features_to_overwrite):
                    raise ValueError('`existing_labels` and `features_to_overwrite` must be of the same length')

                # Overwrite existing features
                if overwrite:
                    existing_ids = [self.mapping.inv[label] for label in existing_labels]
                    self.index.add_items(features_to_overwrite, existing_ids, replace_deleted=True)
                    updated += len(existing_labels)

            incr_size = len(features)
            if incr_size > 0 and len(labels) == incr_size:
                # Check if we need to resize the index
                if self.count + incr_size > self.capacity:
                    self.index.resize_index(self.next_capacity)

                # Prepare IDs and update mapping
                ids = list(range(self.next_id, self.next_id + incr_size))
                for i, label in enumerate(labels):
                    self.mapping[self.next_id + i] = label

                # Add features to index
                self.index.add_items(features, ids, replace_deleted=True)
                updated += incr_size

            if updated:
                self.save()

            return updated

    def get(self, key: int | str) -> Feature:
        """Get feature vector for id or label"""
        return self[key]

    def get_by_ids(self, ids: list[int]) -> list[Feature]:
        """Get feature vectors for multiple ids"""
        try:
            return self.index.get_items(ids).tolist()  # type: ignore
        except RuntimeError as e:
            raise KeyError('Some ids were not found') from e

    def get_by_labels(self, labels: list[str]) -> list[Feature]:
        """Get feature vectors for multiple labels"""
        try:
            ids = [self.mapping.inv[label] for label in labels]
            return self.index.get_items(ids).tolist()  # type: ignore
        except (KeyError, RuntimeError) as e:
            raise KeyError('Some labels were not found') from e

    def has_labels(self, labels: Iterable[str]) -> list[bool]:
        """Check if labels exist in index"""
        return [label in self.mapping.inv for label in labels]

    def save(self):
        """Save database to file"""
        with self.wlock:
            # Save index
            self.index.save_index(str(self.idx_path))

            # Save mapping
            with self.map_path.open('wb') as f:
                dump(self.mapping, f, protocol=HIGHEST_PROTOCOL)

    def delete(self, *keys: int | str, rebuild: bool = False):
        """Delete items from index and rebuild index if needed"""
        with self.wlock:
            for key in keys:
                fid = key if isinstance(key, int) else self.mapping.inv[key]
                self.index.mark_deleted(fid)
                del self.mapping[fid]
            if rebuild:
                self.rebuild(self.dim, self.index.ef_construction, self.index.M)
            self.save()

    def clear(self):
        """Clear database"""
        with self.wlock:
            self.index = self.new_index(dim=self.dim)
            self.mapping = bidict()
            self.save()

    @classmethod
    def drop(cls, db_name: str, base_dir: Path = cfg.BASE_DIR) -> bool:
        """Drop database. Returns True if successful, False otherwise."""
        db_path = base_dir / db_name
        if db_path.exists():
            try:
                shutil.rmtree(db_path)
                return True
            except OSError:
                return False
        return False

    @classmethod
    def db_list(cls, base_dir: Path = cfg.BASE_DIR) -> list[str]:
        """List all available database names in base directory."""
        if not base_dir.exists():
            return []

        databases: list[str] = [
            item.name
            for item in base_dir.iterdir()
            if item.is_dir() and (item / cfg.IDX_NAME).is_file() and (item / cfg.MAP_NAME).is_file()
        ]

        return sorted(databases)

    def rebuild(self, dim: int = 512, ef: int = 400, max_conn: int = 32, n_batch=50000):
        """Rebuild index and mapping"""
        new_index = self.new_index(True, dim, self.capacity, ef, max_conn)
        new_mapping: bidict[int, str] = bidict()

        with self.wlock:
            if self.mapping:
                for n, batch_ids in enumerate(ibatch(self.mapping, n_batch)):
                    # add batch features to new index
                    batch_features = self.index.get_items(batch_ids)
                    new_ids = [n * n_batch + i + 1 for i in range(len(batch_ids))]
                    new_index.add_items(batch_features, new_ids, replace_deleted=True)
                    # update to new mapping
                    batch_labels = [self.mapping[fid] for fid in batch_ids]
                    new_mapping.update(zip(new_ids, batch_labels, strict=True))

            self.index = new_index
            self.mapping = new_mapping
            self.save()

    def search(self, feature: Feature, k: int = 10, similarity: float = 0.0) -> list[tuple[str, float]]:
        """Search items by feature vector with similarity filtering"""
        if self.index is None or self.count == 0 or not feature:
            return []

        # Validate similarity parameter
        if similarity < 0.0 or similarity > 100.0:
            raise ValueError('similarity must be between 0 and 100')

        # Set ef to a value between 150 and 300, depending on the number of results requested.
        # This is to ensure that the search is efficient and fast, without sacrificing accuracy.
        search_k = min(k, self.count)
        self.index.set_ef(min(max(search_k * 3, 150), 300))
        v_ids, distances = self.index.knn_query([feature], k=search_k)

        # Convert results to (label, similarity) tuples
        results = []
        for vid, distance in zip(v_ids[0], distances[0], strict=True):
            if label := self.mapping.get(vid):
                # Convert distance to similarity
                res_similarity = round((1.0 - float(distance)) * 100, 1)
                if res_similarity >= similarity:
                    results.append((label, res_similarity))

        return results
