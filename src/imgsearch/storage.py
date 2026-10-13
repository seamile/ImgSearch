"""A named image table in the local LanceDB root; use one writing service per root."""

import logging
from collections.abc import Iterable, Iterator
from datetime import timedelta
from operator import itemgetter
from pathlib import Path
from threading import RLock
from typing import cast

import lancedb
import numpy as np
import numpy.typing as npt
from lancedb.index import BTree, IvfHnswFlat
from lancedb.query import LanceVectorQueryBuilder

from imgsearch import config as cfg
from imgsearch.models import ImageInfo, ImageModel, SearchResult, as_feature, image_model, search_result


class VectorDB:
    """CRUD and index maintenance for one table in the shared database directory."""

    def __init__(self, db_name: str = cfg.DB_NAME, base_dir: Path = cfg.BASE_DIR, dim: int = 512) -> None:
        """Open or create a named image table and validate its schema."""
        self.model = image_model(dim)
        self.name, self.base, self.dim = db_name, Path(base_dir), dim
        self.path = self.table_path(db_name, self.base)
        legacy = self.base / db_name
        if not self.path.exists() and any(
            (legacy / name).exists() for name in ('images.lance', 'index.db', 'mapping.db')
        ):
            raise ValueError('Legacy database: migrate explicitly with python -m imgsearch.migrate')
        self.wlock = RLock()
        self._changes = 0
        self.connection = lancedb.connect(self.base, read_consistency_interval=timedelta(0))
        if self.path.exists():
            self.table = self.connection.open_table(db_name)
            if not self.table.schema.equals(self.model.to_arrow_schema(), check_metadata=False):
                raise ValueError(f'Table schema or vector dimension does not match {dim}; migration may be required')
        else:
            self.table = self.connection.create_table(db_name, schema=self.model)
            self.table.create_index('label', config=BTree())
            self.table.create_index('id', config=BTree())

    @staticmethod
    def table_path(name: str, base: Path) -> Path:
        """Validate a table name and return its resolved path inside the database directory."""
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Table name must be a directory name')
        path = (Path(base) / f'{name}.lance').resolve()
        if path.parent != Path(base).resolve():
            raise ValueError('Table directory must stay inside base_dir')
        return path

    def __len__(self) -> int:
        """Return the number of stored image records."""
        return self.count

    @staticmethod
    def _condition(keys: Iterable[str], *, by_id: bool = True) -> str:
        """Build an escaped SQL filter for image IDs or labels."""
        values = []
        for key in keys:
            if not isinstance(key, str):
                raise TypeError('Keys must be strings')
            values.append("'" + key.replace("'", "''") + "'")
        field = 'id' if by_id else 'label'
        return f'{field} IN ({",".join(values)})' if values else 'false'

    def __contains__(self, key: str) -> bool:
        """Return whether an image ID exists in the table."""
        with self.wlock:
            return self.table.count_rows(self._condition([key])) > 0

    def __getitem__(self, key: str) -> ImageModel:
        """Return the image record for an ID or raise KeyError if it is missing."""
        rows = self.get_items([key])
        if not rows:
            raise KeyError(key)
        return rows[0]

    @property
    def count(self) -> int:
        """Return the number of rows in the image table."""
        with self.wlock:
            return self.table.count_rows()

    @property
    def size(self) -> int:
        """Disk bytes of this table, including retained versions and indices."""
        with self.wlock:
            return sum(path.stat().st_size for path in self.path.rglob('*') if path.is_file())

    def get_items(self, ids: Iterable[str]) -> list[ImageModel]:
        """Load and validate image records matching the supplied IDs."""
        with self.wlock:
            rows = self.table.search().where(self._condition(ids)).limit(None).to_list()
            return [self.model.model_validate(row) for row in rows]

    def get_info(self, keys: Iterable[str], *, by_id: bool = False) -> list[ImageInfo]:
        """Return metadata sorted by ID, raising KeyError if any requested key is missing."""
        keys = list(keys)
        with self.wlock:
            rows = (
                self.table
                .search()
                .where(self._condition(keys, by_id=by_id))
                .select(['id', 'label', 'paths'])
                .limit(None)
                .to_list()
            )
            found = {row['id' if by_id else 'label'] for row in rows}
            missing = set(keys) - found
            if missing:
                raise KeyError(f'Images not found: {sorted(missing)}')
            return cast(list[ImageInfo], sorted(rows, key=itemgetter('id')))

    def existing_labels(self, labels: Iterable[str]) -> set[str]:
        """Return the supplied labels already stored, without loading vectors or paths."""
        with self.wlock:
            rows = (
                self.table.search().where(self._condition(labels, by_id=False))
                .select(['label']).limit(None).to_list()
            )
            return {row['label'] for row in rows}

    def iter_paths(self) -> Iterator[dict]:
        """Yield image IDs and paths in batches without loading vectors."""
        with self.wlock:
            for batch in self.table.search().select(['id', 'paths']).limit(None).to_batches():
                yield from batch.to_pylist()

    def add_items(self, items: Iterable[ImageModel], *, update_label: bool = False) -> int:
        """Merge records by ID and return the number of inserted or changed records."""
        incoming = {}
        for item in items:
            item = self.model.model_validate(item.model_dump())
            incoming[item.id] = (
                incoming[item.id].merge(item, update_label=update_label) if item.id in incoming else item
            )
        if not incoming:
            return 0
        with self.wlock:
            existing = {item.id: item for item in self.get_items(incoming)}
            rows = []
            for key, item in incoming.items():
                if key in existing:
                    item = existing[key].merge(item, update_label=update_label)
                    if item == existing[key]:
                        continue
                rows.append(item)
            self.replace_items(rows)
            return len(rows)

    def replace_items(self, items: Iterable[ImageModel]) -> None:
        """Commit complete records, including path removals, in one transaction."""
        rows = [self.model.model_validate(item.model_dump()) for item in items]
        if not rows:
            return
        if len({row.id for row in rows}) != len(rows):
            raise ValueError('Replacement records must have unique IDs')
        with self.wlock:
            existing = {item.id: item for item in self.get_items(row.id for row in rows)}
            for row in rows:
                if row.id in existing:
                    existing[row.id].merge(row)  # Detect hash collisions before any write.
            self.table.merge_insert('id').when_matched_update_all().when_not_matched_insert_all().execute(rows)
            self._maintain()

    def delete(self, *keys: str, by_id: bool = False) -> None:
        """Delete records matching the supplied labels or IDs, rejecting missing keys."""
        if not keys:
            return
        with self.wlock:
            rows = self.get_info(keys, by_id=by_id)
            self.table.delete(self._condition(row['id'] for row in rows))
            self._maintain()

    def clear(self) -> None:
        """Remove all records from the table and trigger index maintenance."""
        with self.wlock:
            self.table.delete('true')
            self._maintain()

    def _maintain(self) -> None:
        """Create the vector index when needed and periodically optimize the table."""
        self._changes += 1
        try:
            indices = list(self.table.list_indices())
            if self.count >= cfg.INDEX_MIN_ROWS and not any(index.columns == ['vector'] for index in indices):
                self.table.create_index('vector', config=IvfHnswFlat(distance_type='cosine'))
            if self._changes >= 20:
                self.table.optimize()
                self._changes = 0
        except Exception:
            # The write is committed; maintenance failure must not report data loss.
            logging.getLogger(__name__).exception('LanceDB maintenance failed; writes remain searchable')

    @classmethod
    def drop(cls, db_name: str, base_dir: Path = cfg.BASE_DIR) -> bool:
        """Drop a named table and return whether it existed."""
        path = cls.table_path(db_name, base_dir)
        if not path.is_dir():
            return False
        lancedb.connect(base_dir).drop_table(db_name)
        return True

    @classmethod
    def db_list(cls, base_dir: Path = cfg.BASE_DIR) -> list[str]:
        """Return sorted table names from the database directory."""
        if not Path(base_dir).exists():
            return []
        connection = lancedb.connect(base_dir)
        names, token = [], None
        while True:
            page = connection.list_tables(page_token=token, limit=100)
            names.extend(page.tables)
            token = page.page_token
            if not token:
                return sorted(names)

    def search(self, feature: npt.ArrayLike, k: int = 10, similarity: float = 0.0) -> list[SearchResult]:
        """Return up to k nearest images that meet the minimum cosine similarity score."""
        if not 0.0 <= similarity <= 100.0:
            raise ValueError('similarity must be between 0 and 100')
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        if np.asarray(feature).size == 0:
            return []
        vector = as_feature(feature, self.dim)
        with self.wlock:
            if not self.count:
                return []
            search_k = min(k, self.count)
            ef = max(search_k, min(max(search_k * 3, 150), 300))
            query = cast(LanceVectorQueryBuilder, self.table.search(vector, query_type='vector'))
            rows = (
                query
                .distance_type('cosine')
                .ef(ef)
                .select(['id', 'label', 'paths', '_distance'])
                .limit(search_k)
                .to_list()
            )
            return [hit for row in rows if (hit := search_result(row))['score'] >= similarity]
