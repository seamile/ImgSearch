"""Exercise model identity and real LanceDB table operations."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import xxhash
from lancedb.index import IvfHnswFlat

from imgsearch import config as cfg
from imgsearch.models import image_model
from imgsearch.storage import VectorDB


@pytest.fixture
def db(tmp_path):
    return VectorDB('test', tmp_path, dim=3)


def add(db, labels, features):
    return db.add_items(db.model.from_feature(v, label=label) for label, v in zip(labels, features, strict=True))


@pytest.mark.parametrize('dim', [512, 1024])
def test_dimensions_identity_and_persistence(tmp_path, dim):
    db = VectorDB('test', tmp_path, dim)
    vector = np.arange(dim, dtype=np.float32)
    row = db.model.from_path(vector, tmp_path / 'image.jpg')
    assert row.id == xxhash.xxh64_hexdigest(vector.tobytes())
    assert db.add_items([row]) == 1
    assert db.path == tmp_path / 'test.lance' and db.size > 0
    reopened = VectorDB('test', tmp_path, dim)
    assert reopened[row.id] == row
    assert np.asarray(reopened[row.id].vector, dtype='<f4').tobytes() == vector.tobytes()
    hit = reopened.search(vector)[0]
    assert hit == {'id': row.id, 'label': 'image', 'paths': [str(tmp_path / 'image.jpg')], 'score': 100.}
    with pytest.raises(ValueError, match='dimension'):
        VectorDB('test', tmp_path, dim + 1)


def test_paths_and_update_label(db, tmp_path):
    first = db.model.from_path([1, 0, 0], tmp_path / 'first.jpg')
    second = db.model.from_path([1, 0, 0], tmp_path / 'second.jpg')
    assert db.add_items([first, second, first]) == 1
    assert db[first.id].label == 'first'
    assert db[first.id].paths == first.paths + second.paths
    assert db.add_items([second]) == 0
    assert db.add_items([second], update_label=True) == 1
    assert db[first.id].label == 'second'
    assert db.add_items([first, second], update_label=True) == 0
    assert db.add_items([]) == 0
    assert db.count == 1
    assert first.paths == [str(tmp_path / 'first.jpg')]


def test_same_label_and_changed_path_keep_distinct_ids(db, tmp_path):
    path = tmp_path / 'same.jpg'
    a, b = [db.model.from_path(v, path) for v in ([1, 0, 0], [0, 1, 0])]
    db.add_items([a, b])
    assert len(db.get_info(['same'])) == 2
    assert db[a.id].paths == db[b.id].paths == [str(path)]


def test_model_validation_and_canonical_bytes():
    model = image_model(3)
    a = model.from_feature(np.array([3, 0, 0], dtype='>f4'), label='a')
    b = model.from_feature([3, 0, 0], label='b')
    assert a.id == b.id and a.paths == [] and a.vector[0] == 3
    assert model is image_model(3)
    with pytest.raises(ValueError):
        model(id=a.id, label='a', vector=[0, 1, 0])
    with pytest.raises(ValueError):
        model(id=a.id, label='a', vector=a.vector, paths=None)
    with pytest.raises(ValueError):
        model.from_feature(a.vector, label='')
    with pytest.raises(ValueError):
        model.from_feature(a.vector, label='a', paths=[''])


@pytest.mark.parametrize('feature', [[1., 0.], [0., 0., 0.], [float('nan'), 0., 1.], [float('inf'), 0., 1.]])
def test_invalid_vectors_do_not_write(db, feature):
    with pytest.raises(ValueError):
        add(db, ['valid', 'bad'], [[1, 0, 0], feature])
    assert db.count == 0
    with pytest.raises(ValueError):
        db.search(feature)


def test_hash_collision_does_not_overwrite(db):
    with patch('imgsearch.models.xxhash.xxh64_hexdigest', return_value='0' * 16):
        a = db.model.from_feature([1, 0, 0], label='a')
        b = db.model.from_feature([0, 1, 0], label='b')
        db.add_items([a])
        with pytest.raises(ValueError, match='Conflicting'):
            db.add_items([b])
        assert db[a.id].label == 'a' and db.count == 1


def test_deletion_and_sql_escaping(db):
    labels = ["it's.jpg", "x' OR true --", '你好.jpg']
    add(db, labels, np.eye(3))
    rows = db.get_info(labels)
    assert len(rows) == 3 and db.get_info([]) == []
    assert rows[0]['id'] in db and 'missing' not in db
    with pytest.raises(KeyError):
        db.delete(labels[0], 'missing')
    assert db.count == 3
    db.delete(labels[1])
    assert db.count == 2
    db.delete(rows[0]['id'], by_id=True)
    assert db.count == 1
    with pytest.raises(KeyError):
        db['missing']
    with pytest.raises(TypeError):
        db[1]
    db.delete()
    db.clear()
    assert db.count == 0


def test_cosine_and_validation(db):
    add(db, ['same', 'angled', 'orthogonal', 'opposite'], [
        [3., 0., 0.], [0.7996, (1 - 0.7996**2)**0.5, 0.], [0., 1., 0.], [-1., 0., 0.],
    ])
    hits = db.search(np.array([2., 0., 0.]), k=20)
    assert [(hit['label'], hit['score']) for hit in hits] == [('same', 100.), ('angled', 80.), ('orthogonal', 0.)]
    assert len(db.search([1, 0, 0], similarity=80)) == 2
    assert db.search([]) == []
    for similarity in [-1, 101, float('nan')]:
        with pytest.raises(ValueError):
            db.search([1, 0, 0], similarity=similarity)
    for k in [0, -1, 1.5]:
        with pytest.raises(ValueError):
            db.search([1, 0, 0], k=k)


def test_root_layout_drop_and_legacy_detection(tmp_path):
    assert VectorDB.db_list(tmp_path / 'missing') == []
    a = VectorDB('a', tmp_path)
    b = VectorDB('b', tmp_path)
    assert VectorDB.db_list(tmp_path) == ['a', 'b']
    (tmp_path / 'service.log').write_text('keep')
    a.clear()
    assert VectorDB.drop('a', tmp_path)
    assert b.path.is_dir() and (tmp_path / 'service.log').read_text() == 'keep'
    assert not VectorDB.drop('missing', tmp_path)
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    (legacy / 'images.lance').mkdir()
    with pytest.raises(ValueError, match='Legacy'):
        VectorDB('legacy', tmp_path)
    assert not (tmp_path / 'legacy.lance').exists()
    for name in ['../outside', '.', '..', str(tmp_path / 'unsafe'), '']:
        with pytest.raises(ValueError):
            VectorDB(name, tmp_path)
        with pytest.raises(ValueError):
            VectorDB.drop(name, tmp_path)
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    (tmp_path / 'link.lance').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='inside base_dir'):
        VectorDB('link', tmp_path)
    with pytest.raises(ValueError):
        VectorDB('bad', tmp_path, dim=0)


def test_index_mutations_and_maintenance(db, monkeypatch, caplog):
    monkeypatch.setattr(cfg, 'INDEX_MIN_ROWS', 256)
    rng = np.random.default_rng(42)
    add(db, [str(i) for i in range(256)], rng.normal(size=(256, 3)))
    assert any(index.columns == ['vector'] for index in db.table.list_indices())
    add(db, ['new'], [[1, 0, 0]])
    assert db.search([1, 0, 0], k=1)[0]['label'] == 'new'
    db.delete('new')
    assert all(hit['label'] != 'new' for hit in db.search([1, 0, 0], k=257))
    db._changes = 19
    with patch.object(db.table, 'optimize', side_effect=RuntimeError('maintenance')):
        add(db, ['after'], [[1, 0, 0]])
    assert 'maintenance failed' in caplog.text
    assert db.search([1, 0, 0], k=1)[0]['label'] == 'after'
    db.clear()
    assert db.search([1, 0, 0]) == []


def test_ann_recall_against_exact_cosine(tmp_path):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(1500, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(32, 32)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    db = VectorDB('ann', tmp_path, dim=32)
    add(db, [str(i) for i in range(len(vectors))], vectors)
    db.table.create_index('vector', config=IvfHnswFlat(
        distance_type='cosine', num_partitions=1, m=8, ef_construction=64,
    ))
    recalls = []
    for query in queries:
        expected = set(np.argpartition(-(vectors @ query), 9)[:10])
        found = {int(row['label']) for row in db.search(query, k=10)}
        recalls.append(len(expected & found) / 10)
    assert np.mean(recalls) >= 0.97
    assert len(db.search(queries[0], k=500)) == 500


def test_concurrent_merges_keep_all_paths(db, tmp_path):
    def operation(i):
        row = db.model.from_path([1, 0, 0], tmp_path / f'{i}.jpg')
        db.add_items([row])
        assert db.search([1, 0, 0])
        return row.id
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(operation, range(12)))
    assert len(set(ids)) == db.count == 1
    assert len(db[ids[0]].paths) == 12


def test_table_listing_pagination(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    connection = Mock()
    next_page = 'next'
    connection.list_tables.side_effect = [
        SimpleNamespace(tables=['b'], page_token=next_page),
        SimpleNamespace(tables=['a'], page_token=None),
    ]
    with patch('imgsearch.storage.lancedb.connect', return_value=connection):
        assert VectorDB.db_list(tmp_path) == ['a', 'b']
    connection.list_tables.assert_called_with(page_token=next_page, limit=100)
