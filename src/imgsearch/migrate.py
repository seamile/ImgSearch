"""Migrate a trusted HNSW or legacy LanceDB directory to a named image table."""

import argparse
from pathlib import Path
from pickle import Unpickler
from tempfile import TemporaryDirectory

import lancedb

from imgsearch.storage import VectorDB


class _MappingUnpickler(Unpickler):
    def find_class(self, module, name):
        # Historical bidict reducer used a different constructor signature.
        """Resolve pickle classes while adapting the historical bidict reconstruction helper."""
        if module == 'bidict._base' and name == 'BidictBase._from_other':
            return lambda cls, items, inverted: {v: k for k, v in items.items()} if inverted else dict(items)
        return super().find_class(module, name)


def _legacy_rows(source: Path, dim: int):
    """Read legacy LanceDB or HNSW data as a vector dimension and iterable row batches."""
    if (source / 'images.lance').is_dir():
        table = lancedb.connect(source).open_table('images')
        if table.schema.names != ['id', 'label', 'vector']:
            raise ValueError('Expected legacy id/label/vector schema')
        dim = table.schema.field('vector').type.list_size
        # Stable legacy ID order determines the retained label for duplicates.
        rows = table.to_arrow().sort_by([('id', 'ascending')]).to_batches(max_chunksize=5000)
        return dim, (batch.to_pylist() for batch in rows)
    if not (source / 'index.db').is_file() or not (source / 'mapping.db').is_file():
        raise ValueError('Source must contain images.lance or index.db and mapping.db')
    from hnswlib import Index

    # Pickle is accepted only for trusted local backups.
    with (source / 'mapping.db').open('rb') as stream:
        mapping = dict(_MappingUnpickler(stream).load())
    if any(not isinstance(key, int) or key <= 0 for key in mapping):
        raise ValueError('Legacy IDs must be positive integers')
    if any(not isinstance(label, str) or not label for label in mapping.values()):
        raise ValueError('Legacy labels must be nonempty strings')
    index = Index(space='cosine', dim=dim)
    index.load_index(str(source / 'index.db'))
    if not set(mapping).issubset(index.get_ids_list()):
        raise ValueError('Index and mapping files are not consistent')
    ids = sorted(mapping)

    def batches():
        """Yield legacy HNSW vectors and labels in batches ordered by ID."""
        for offset in range(0, len(ids), 5000):
            keys = ids[offset : offset + 5000]
            vectors = index.get_items(keys)
            yield [
                {'id': key, 'label': mapping[key], 'vector': vector} for key, vector in zip(keys, vectors, strict=True)
            ]

    return dim, batches()


def migrate(source: Path, db_path: Path, dim: int = 512, *, table_name: str = 'default') -> int:
    """Publish a validated new table; never modify the source or replace a table."""
    source, db_path = Path(source).resolve(), Path(db_path).resolve()
    target = VectorDB.table_path(table_name, db_path)
    if target.exists() or source == db_path or source in db_path.parents:
        raise ValueError('Destination must be outside the source with a new table name')
    dim, batches = _legacy_rows(source, dim)
    db_path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.isearch-migrate-', dir=db_path) as temporary:
        db = VectorDB(table_name, Path(temporary), dim)
        expected = set()
        for batch in batches:
            records = []
            for row in batch:
                label = row['label']
                if not isinstance(label, str) or not label:
                    raise ValueError('Legacy labels must be nonempty strings')
                if Path(label).is_absolute():
                    item = db.model.from_path(row['vector'], label)
                else:
                    item = db.model.from_feature(row['vector'], label=label)
                records.append(item)
                expected.add(item.id)
            db.add_items(records)
        if db.count != len(expected):
            raise ValueError('Migrated row count does not match unique vector IDs')
        for batch in db.table.search().limit(None).to_batches():
            for row in batch.to_pylist():
                db.model.model_validate(row)
        if target.exists():
            raise ValueError('Destination table appeared during migration')
        db.path.rename(target)
    return len(expected)


def main() -> None:
    """Parse migration arguments and report the number of migrated image records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Trusted legacy directory; stop the service first')
    parser.add_argument('db_path', type=Path, help='LanceDB root directory')
    parser.add_argument('--table', dest='table_name', default='default', help='New table name')
    parser.add_argument('--dim', type=int, default=512, help='HNSW vector dimension; inferred for LanceDB')
    args = parser.parse_args()
    count = migrate(args.source, args.db_path, args.dim, table_name=args.table_name)
    print(f'Migrated {count} unique records to {args.db_path / (args.table_name + ".lance")}; source preserved.')


if __name__ == '__main__':
    main()
