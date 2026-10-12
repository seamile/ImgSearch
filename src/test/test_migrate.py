"""Migration preserves source data and publishes hash-based tables only on success."""

from pickle import dump

import pytest

from imgsearch.migrate import migrate
from imgsearch.storage import VectorDB


@pytest.fixture
def legacy(tmp_path):
    hnswlib = pytest.importorskip('hnswlib')
    source = tmp_path / 'old'
    source.mkdir()
    index = hnswlib.Index(space='cosine', dim=3)
    index.init_index(max_elements=10)
    index.add_items([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [2, 7, 9])
    index.mark_deleted(9)
    index.save_index(str(source / 'index.db'))
    with (source / 'mapping.db').open('wb') as stream:
        dump({2: 'a', 7: 'b'}, stream)
    return source


def test_migrate_active_ids_and_vectors(legacy, tmp_path):
    original = {path.name: path.read_bytes() for path in legacy.iterdir()}
    target = tmp_path / 'new'
    assert migrate(legacy, target, dim=3) == 2
    assert original == {path.name: path.read_bytes() for path in legacy.iterdir()}
    db = VectorDB('default', tmp_path / 'new', dim=3)
    assert db.count == 2
    assert [hit['label'] for hit in db.search([1., 0., 0.])] == ['a', 'b']
    row = db.model.from_feature([0., 0., 1.], label='c')
    db.add_items([row])
    assert row.id in db


def test_migration_failure_does_not_publish_partial_data(legacy, tmp_path):
    with (legacy / 'mapping.db').open('wb') as stream:
        dump({2: 'a', 99: 'missing'}, stream)
    target = tmp_path / 'new'
    with pytest.raises(ValueError, match='consistent'):
        migrate(legacy, target, dim=3)
    assert not (target / 'default.lance').exists()
    assert not list(tmp_path.glob('.isearch-migrate-*'))


def test_deleted_mapped_item_aborts_export(legacy, tmp_path):
    with (legacy / 'mapping.db').open('wb') as stream:
        dump({2: 'a', 9: 'deleted'}, stream)
    with pytest.raises(RuntimeError):
        migrate(legacy, tmp_path / 'new', dim=3)
    assert not (tmp_path / 'new' / 'default.lance').exists()
    assert not list(tmp_path.glob('.isearch-migrate-*'))


def test_migration_rejects_overwrite_and_nested_target(legacy, tmp_path):
    VectorDB('default', tmp_path, dim=3)
    for destination in [legacy, legacy / 'nested', tmp_path]:
        with pytest.raises(ValueError, match='Destination'):
            migrate(legacy, destination, dim=3)


def test_empty_legacy_database(tmp_path):
    hnswlib = pytest.importorskip('hnswlib')
    source = tmp_path / 'old'
    source.mkdir()
    index = hnswlib.Index(space='cosine', dim=3)
    index.init_index(max_elements=10)
    index.save_index(str(source / 'index.db'))
    with (source / 'mapping.db').open('wb') as stream:
        dump({}, stream)
    assert migrate(source, tmp_path / 'new', dim=3) == 0
    assert VectorDB('default', tmp_path / 'new', dim=3).count == 0


def test_old_bidict_pickle(legacy, tmp_path):
    pytest.importorskip('bidict')
    # Historical global path and (class, items, inverted) reducer arguments.
    (legacy / 'mapping.db').write_bytes(
        b'cbidict._base\nBidictBase._from_other\n(cbidict\nbidict\n(dp0\nI2\nVa\np1\nsI7\nVb\np2\nsI00\ntR.'
    )
    assert migrate(legacy, tmp_path / 'new', dim=3) == 2
    db = VectorDB('default', tmp_path / 'new', dim=3)
    assert db.get_items([db.model.from_feature([1., 0., 0.], label='a').id])[0].label == 'a'
    assert db.get_items([db.model.from_feature([0., 1., 0.], label='b').id])[0].label == 'b'


def test_current_bidict_pickle(legacy, tmp_path):
    bidict = pytest.importorskip('bidict').bidict
    with (legacy / 'mapping.db').open('wb') as stream:
        dump(bidict({2: 'a', 7: 'b'}), stream)
    assert migrate(legacy, tmp_path / 'new', dim=3) == 2


def test_lance_migration_merges_paths_and_preserves_vectors(tmp_path):
    import lancedb
    import numpy as np
    import pyarrow as pa
    source = tmp_path / 'old'
    schema = pa.schema([('id', pa.int64()), ('label', pa.string()), ('vector', pa.list_(pa.float32(), 3))])
    old = lancedb.connect(source).create_table('images', schema=schema)
    old.add([
        {'id': 8, 'label': '/missing/second.png', 'vector': [2., 0., 0.]},
        {'id': 2, 'label': '/missing/first.jpg', 'vector': [2., 0., 0.]},
        {'id': 9, 'label': 'name.with.dot', 'vector': [0., 1., 0.]},
    ])
    before = old.to_arrow().to_pylist()
    assert migrate(source, tmp_path, table_name='old') == 2
    db = VectorDB('old', tmp_path, 3)
    row = db.model.from_feature([2, 0, 0], label='first')
    actual = db[row.id]
    assert actual.label == 'first' and actual.paths == ['/missing/first.jpg', '/missing/second.png']
    assert np.asarray(actual.vector, dtype=np.float32).tobytes() == np.array([2, 0, 0], dtype=np.float32).tobytes()
    assert db.get_info(['name.with.dot'])[0]['paths'] == []
    assert old.to_arrow().to_pylist() == before
    with pytest.raises(ValueError, match='Destination'):
        migrate(source, tmp_path, table_name='old')


def test_lance_migration_failure_keeps_source(tmp_path):
    import lancedb
    import pyarrow as pa
    source = tmp_path / 'old'
    old = lancedb.connect(source).create_table('images', schema=pa.schema([
        ('id', pa.int64()), ('label', pa.string()), ('vector', pa.list_(pa.float32(), 3)),
    ]))
    old.add([{'id': 1, 'label': 'a', 'vector': [0., 0., 0.]}])
    with pytest.raises(ValueError):
        migrate(source, tmp_path, table_name='new')
    assert not (tmp_path / 'new.lance').exists()
    assert old.count_rows() == 1
    assert not list(tmp_path.glob('.isearch-migrate-*'))


def test_migrated_index_reopens_after_directory_publish(tmp_path, monkeypatch):
    import lancedb
    import numpy as np
    import pyarrow as pa

    from imgsearch import config as cfg
    monkeypatch.setattr(cfg, 'INDEX_MIN_ROWS', 256)
    source = tmp_path / 'old'
    old = lancedb.connect(source).create_table('images', schema=pa.schema([
        ('id', pa.int64()), ('label', pa.string()), ('vector', pa.list_(pa.float32(), 3)),
    ]))
    vectors = np.random.default_rng(42).normal(size=(256, 3)).astype(np.float32)
    old.add([{'id': i + 1, 'label': str(i), 'vector': vector.tolist()} for i, vector in enumerate(vectors)])
    assert migrate(source, tmp_path, table_name='new') == 256
    db = VectorDB('new', tmp_path, 3)
    assert any(index.columns == ['vector'] for index in db.table.list_indices())
    assert db.search(vectors[0], k=1)[0]['label'] == '0'
