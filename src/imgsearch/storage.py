"""Local LanceDB storage. Each database has one table; use one writer per database."""

import logging
import shutil
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from threading import RLock
from typing import cast

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import BTree, IvfHnswFlat
from lancedb.query import LanceVectorQueryBuilder

from imgsearch import config as cfg
from imgsearch.utils import Feature


class VectorDB:
    """Persistent vectors keyed by label and integer ID, searched with cosine distance."""

    def __init__(self, db_name: str = cfg.DB_NAME, base_dir: Path = cfg.BASE_DIR, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError('dim must be positive')
        if not db_name or Path(db_name).name != db_name or db_name in {'.', '..'}:
            raise ValueError('db_name must be a directory name')
        self.name = db_name
        self.base = Path(base_dir)
        self.dim = dim
        self.path = (self.base / db_name).resolve()
        if not self.path.is_relative_to(self.base.resolve()):
            raise ValueError('Database directory must stay inside base_dir')
        # Never silently open an old database as an empty Lance table.
        if any((self.path / name).exists() for name in ('index.db', 'mapping.db')):
            raise ValueError('Legacy HNSW database: migrate to a new directory with python -m imgsearch.migrate')
        self.wlock = RLock()
        self._changes = 0
        self.connection = lancedb.connect(self.path, read_consistency_interval=timedelta(0))
        table_path = self.path / f'{cfg.TABLE_NAME}.lance'
        if table_path.exists():
            self.table = self.connection.open_table(cfg.TABLE_NAME)
            if self.table.schema.field('vector').type != pa.list_(pa.float32(), dim):
                raise ValueError(f'Database vector dimension does not match {dim}')
        else:
            schema = pa.schema([
                pa.field('id', pa.int64(), metadata={'next_id': '1'}),
                pa.field('label', pa.string()),
                pa.field('vector', pa.list_(pa.float32(), dim)),
            ])
            self.table = self.connection.create_table(cfg.TABLE_NAME, schema=schema)
            self.table.create_index('label', config=BTree())
            self.table.create_index('id', config=BTree())

    def __len__(self) -> int:
        return self.count

    @staticmethod
    def _condition(keys: Iterable[int | str]) -> str:
        ids, labels = [], []
        for key in keys:
            if isinstance(key, int):
                ids.append(str(key))
            elif isinstance(key, str):
                labels.append("'" + key.replace("'", "''") + "'")
            else:
                raise TypeError('Keys must be integer IDs or string labels')
        terms = []
        if ids:
            terms.append('id IN (' + ','.join(ids) + ')')
        if labels:
            terms.append('label IN (' + ','.join(labels) + ')')
        return ' OR '.join(terms) or 'false'

    def __contains__(self, key: int | str) -> bool:
        with self.wlock:
            return self.table.count_rows(self._condition([key])) > 0

    def __getitem__(self, key: int | str) -> Feature:
        return self._get_many([key])[0]

    @property
    def count(self) -> int:
        with self.wlock:
            return self.table.count_rows()

    @property
    def size(self) -> int:
        """Total disk bytes, including retained versions and index files."""
        with self.wlock:
            return sum(path.stat().st_size for path in self.path.rglob('*') if path.is_file())

    @property
    def next_id(self) -> int:
        with self.wlock:
            return int(self.table.schema.field('id').metadata[b'next_id'])

    def _feature(self, feature: Feature) -> Feature:
        vector = np.asarray(feature, dtype=np.float32)
        if vector.shape != (self.dim,) or not np.isfinite(vector).all():
            raise ValueError(f'Feature must contain {self.dim} finite numbers')
        norm = np.linalg.norm(vector.astype(np.float64))
        if norm == 0:
            raise ValueError('Cosine features must be nonzero')
        return (vector / norm).astype(np.float32).tolist()

    def add_items(self, labels: list[str], features: list[Feature], *, overwrite: bool = True) -> int:  # ruff: ignore[complex-structure]
        if len(labels) != len(features):
            raise ValueError('Labels and features must be of the same length')
        incoming = {}
        for label, feature in zip(labels, features, strict=True):
            if not isinstance(label, str) or not label:
                raise ValueError('Labels must be nonempty strings')
            vector = self._feature(feature)
            if overwrite or label not in incoming:
                incoming[label] = vector
        if not incoming:
            return 0
        with self.wlock:
            existing = {
                row['label']: row['id']
                for row in self.table
                .search()
                .where(self._condition(incoming))
                .select(['id', 'label'])
                .limit(None)
                .to_list()
            }
            next_id = self.next_id
            rows = []
            for label, vector in incoming.items():
                if label in existing:
                    if not overwrite:
                        continue
                    fid = existing[label]
                else:
                    fid = next_id
                    next_id += 1
                rows.append({'id': fid, 'label': label, 'vector': vector})
            if not rows:
                return 0
            if next_id != self.next_id:
                # Reserve IDs before committing so deletes/restarts never reuse them.
                self.table.update_field_metadata({'path': 'id', 'metadata': {'next_id': str(next_id)}})
            self.table.merge_insert('label').when_matched_update_all().when_not_matched_insert_all().execute(rows)
            self._maintain()
            return len(rows)

    def _get_many(self, keys: list[int | str]) -> list[Feature]:
        if not keys:
            return []
        with self.wlock:
            rows = self.table.search().where(self._condition(keys)).limit(None).to_list()
            by_id = {row['id']: row['vector'] for row in rows}
            by_label = {row['label']: row['vector'] for row in rows}
            return [(by_id if isinstance(key, int) else by_label)[key] for key in keys]

    def has_labels(self, labels: Iterable[str]) -> list[bool]:
        labels = list(labels)
        with self.wlock:
            found = {
                row['label']
                for row in self.table.search().where(self._condition(labels)).select(['label']).limit(None).to_list()
            }
            return [label in found for label in labels]

    def delete(self, *keys: int | str) -> None:
        if not keys:
            return
        with self.wlock:
            self._get_many(list(keys))  # Validate all keys before deleting any rows.
            self.table.delete(self._condition(keys))
            self._maintain()

    def clear(self) -> None:
        with self.wlock:
            self.table.delete('true')
            self._maintain()

    def _maintain(self) -> None:
        self._changes += 1
        try:
            indices = list(self.table.list_indices())
            if self.count >= cfg.INDEX_MIN_ROWS and not any(index.columns == ['vector'] for index in indices):
                self.table.create_index('vector', config=IvfHnswFlat(distance_type='cosine'))
            if self._changes >= 20:
                self.table.optimize()
                self._changes = 0
        except Exception:
            # The write is already committed; maintenance failure must not report data loss.
            logging.getLogger(__name__).exception('LanceDB maintenance failed; writes remain searchable')

    @classmethod
    def drop(cls, db_name: str, base_dir: Path = cfg.BASE_DIR) -> bool:
        if not db_name or Path(db_name).name != db_name or db_name in {'.', '..'}:
            raise ValueError('db_name must be a directory name')
        path = Path(base_dir) / db_name
        if path.is_dir():
            try:
                shutil.rmtree(path)
                return True
            except OSError:
                return False
        return False

    @classmethod
    def db_list(cls, base_dir: Path = cfg.BASE_DIR) -> list[str]:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            return []
        return sorted(path.name for path in base_dir.iterdir() if (path / f'{cfg.TABLE_NAME}.lance').is_dir())

    def search(self, feature: Feature, k: int = 10, similarity: float = 0.0) -> list[tuple[str, float]]:
        if not 0.0 <= similarity <= 100.0:
            raise ValueError('similarity must be between 0 and 100')
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        if not feature:
            return []
        vector = self._feature(feature)
        with self.wlock:
            if not self.count:
                return []
            search_k = min(k, self.count)
            # Match the previous HNSW search effort; Lance's default ef is only 1.5 * k.
            ef = max(search_k, min(max(search_k * 3, 150), 300))
            query = cast(LanceVectorQueryBuilder, self.table.search(vector, query_type='vector'))
            rows = query.distance_type('cosine').ef(ef).limit(search_k).to_list()
            results = []
            for row in rows:
                score = round((1.0 - float(row['_distance'])) * 100, 1)
                if score >= similarity:
                    results.append((row['label'], score))
            return results
