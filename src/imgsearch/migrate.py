"""Export a trusted local HNSW database into a separate LanceDB directory."""

import argparse
from pathlib import Path
from pickle import Unpickler
from tempfile import TemporaryDirectory

from imgsearch.storage import VectorDB


class _MappingUnpickler(Unpickler):
    def find_class(self, module, name):
        # Older bidict pickles pass (class, items, inverted) to a now-changed constructor.
        if module == 'bidict._base' and name == 'BidictBase._from_other':
            return lambda cls, items, inverted: {v: k for k, v in items.items()} if inverted else dict(items)
        return super().find_class(module, name)


def migrate(source: Path, destination: Path, dim: int = 512) -> int:
    """Keep active IDs/labels and vectors; leave the source intact on any failure."""
    from hnswlib import Index

    source, destination = source.resolve(), destination.absolute()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError('Destination must be a new directory outside the source database')
    if not (source / 'index.db').is_file() or not (source / 'mapping.db').is_file():
        raise ValueError('Source must contain index.db and mapping.db')
    # Only run this tool on trusted local backups: pickle can execute code.
    with (source / 'mapping.db').open('rb') as stream:
        mapping = dict(_MappingUnpickler(stream).load())
    if any(not isinstance(fid, int) or fid <= 0 for fid in mapping):
        raise ValueError('Legacy IDs must be positive integers')
    if any(not isinstance(label, str) or not label for label in mapping.values()):
        raise ValueError('Legacy labels must be nonempty strings')
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Legacy labels must be unique')
    index = Index(space='cosine', dim=dim)
    index.load_index(str(source / 'index.db'))
    stored_ids = index.get_ids_list()
    if not set(mapping).issubset(stored_ids):
        raise ValueError('Index and mapping files are not consistent')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.isearch-migrate-', dir=destination.parent) as temporary:
        db = VectorDB('database', Path(temporary), dim)
        ids = list(mapping)
        for offset in range(0, len(ids), 5000):
            batch = ids[offset : offset + 5000]
            vectors = index.get_items(batch)
            db.table.add([
                {'id': fid, 'label': mapping[fid], 'vector': db._feature(vector.tolist())}
                for fid, vector in zip(batch, vectors, strict=True)
            ])
        next_id = max(stored_ids, default=0) + 1
        db.table.update_field_metadata({'path': 'id', 'metadata': {'next_id': str(next_id)}})
        if db.count != len(mapping):
            raise ValueError('Migrated row count does not match mapping')
        db._maintain()
        # The destination only appears after the complete export has succeeded.
        db.path.rename(destination)
    return len(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Trusted HNSW database directory; stop the service first')
    parser.add_argument('destination', type=Path, help='New LanceDB directory (must not exist)')
    parser.add_argument('--dim', type=int, default=512, help='Original model vector dimension')
    args = parser.parse_args()
    count = migrate(args.source, args.destination, args.dim)
    print(f'Migrated {count} items to {args.destination}; source was preserved.')


if __name__ == '__main__':
    main()
