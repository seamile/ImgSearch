"""Migration reads only live labels, preserves IDs, and never changes the source."""

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
    db = VectorDB('new', tmp_path, dim=3)
    assert db.count == 2 and db.next_id == 10
    assert db[2] == [1., 0., 0.]
    assert db[7] == [0., 1., 0.]
    assert 9 not in db
    assert db.search([1., 0., 0.]) == [('a', 100.), ('b', 0.)]
    db.add_items(['c'], [[0., 0., 1.]])
    assert 10 in db


def test_migration_failure_does_not_publish_partial_data(legacy, tmp_path):
    with (legacy / 'mapping.db').open('wb') as stream:
        dump({2: 'a', 99: 'missing'}, stream)
    target = tmp_path / 'new'
    with pytest.raises(ValueError, match='consistent'):
        migrate(legacy, target, dim=3)
    assert not target.exists()
    assert not list(tmp_path.glob('.isearch-migrate-*'))


def test_deleted_mapped_item_aborts_export(legacy, tmp_path):
    with (legacy / 'mapping.db').open('wb') as stream:
        dump({2: 'a', 9: 'deleted'}, stream)
    with pytest.raises(RuntimeError):
        migrate(legacy, tmp_path / 'new', dim=3)
    assert not (tmp_path / 'new').exists()
    assert not list(tmp_path.glob('.isearch-migrate-*'))


def test_migration_rejects_overwrite_and_nested_target(legacy, tmp_path):
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
    assert VectorDB('new', tmp_path, dim=3).count == 0


def test_old_bidict_pickle(legacy, tmp_path):
    pytest.importorskip('bidict')
    # Historical global path and (class, items, inverted) reducer arguments.
    (legacy / 'mapping.db').write_bytes(
        b'cbidict._base\nBidictBase._from_other\n(cbidict\nbidict\n(dp0\nI2\nVa\np1\nsI7\nVb\np2\nsI00\ntR.'
    )
    assert migrate(legacy, tmp_path / 'new', dim=3) == 2
    db = VectorDB('new', tmp_path, dim=3)
    assert db[2] == [1., 0., 0.]
    assert db[7] == [0., 1., 0.]


def test_current_bidict_pickle(legacy, tmp_path):
    bidict = pytest.importorskip('bidict').bidict
    with (legacy / 'mapping.db').open('wb') as stream:
        dump(bidict({2: 'a', 7: 'b'}), stream)
    assert migrate(legacy, tmp_path / 'new', dim=3) == 2
