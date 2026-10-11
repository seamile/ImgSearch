"""Behavior tests against the installed LanceDB, including its actual indices."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from lancedb.index import IvfHnswFlat

from imgsearch import config as cfg
from imgsearch.storage import VectorDB


@pytest.fixture
def db(tmp_path):
    return VectorDB('test', tmp_path, dim=3)


@pytest.mark.parametrize('dim', [512, 1024])
def test_dimensions_and_persistence(tmp_path, dim):
    db = VectorDB('test', tmp_path, dim)
    feature = [1.0] + [0.0] * (dim - 1)
    assert len(db) == 0
    assert db.search(feature) == []
    assert db.add_items(['image'], [feature])
    assert db.size > 0
    reopened = VectorDB('test', tmp_path, dim)
    assert reopened.count == 1
    assert reopened[1] == feature
    assert reopened['image'] == feature
    assert reopened.search(feature) == [('image', 100.0)]
    with pytest.raises(ValueError, match='dimension'):
        VectorDB('test', tmp_path, dim + 1)


def test_overwrite_order_and_duplicates(db):
    labels = ['z', 'a', 'z']
    features = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    assert db.add_items(labels, features) == 2
    assert labels == ['z', 'a', 'z']  # Caller inputs are not modified.
    assert len(features) == 3
    assert db[1] == features[2]
    assert db[2] == features[1]
    assert not db.add_items(['z'], [features[0]], overwrite=False)
    assert db.add_items(['z', 'a'], [features[0], features[2]]) == 2
    assert db[2] == features[2]
    assert db[1] == features[0]
    assert db['a'] == features[2]
    assert db['z'] == features[0]
    assert db.next_id == 3
    assert db.add_items(['b', 'b'], [features[0], features[2]], overwrite=False) == 1
    assert db['b'] == features[0]
    assert db.add_items([], []) == 0
    assert db.has_labels([]) == []


def test_keys_and_sql_escaping(db):
    labels = ["it's.jpg", "x' OR true --", r'C:\photos\a.jpg', '你好.jpg']
    db.add_items(labels, [[1., 0., 0.]] * len(labels))
    assert db.has_labels(labels + ['missing', labels[0]]) == [True] * 4 + [False, True]
    assert 1 in db and labels[0] in db
    assert 99 not in db and 'missing' not in db
    assert db[labels[1]] == [1., 0., 0.]
    db.delete(labels[1])
    assert db.count == 3
    for key in [99, 'missing']:
        with pytest.raises(KeyError):
            db[key]
    with pytest.raises(TypeError):
        db[None]


def test_delete_atomic_clear_and_no_id_reuse(db):
    db.add_items(['a', 'b'], [[1., 0., 0.], [0., 1., 0.]])
    with pytest.raises(KeyError):
        db.delete('a', 'missing')
    assert db.count == 2
    db.delete('a', 2)
    assert db.count == 0
    reopened = VectorDB(db.name, db.base, db.dim)
    assert reopened.count == 0
    assert reopened.add_items(['c'], [[1., 0., 0.]])
    assert 3 in reopened and 1 not in reopened
    reopened.clear()
    assert reopened.count == 0
    reopened.add_items(['d'], [[0., 1., 0.]])
    assert 4 in reopened
    reopened.delete()
    assert reopened.count == 1


def test_cosine_threshold_and_rounding(db):
    db.add_items(['same', 'angled', 'orthogonal', 'opposite'], [
        [3., 0., 0.], [0.7996, (1 - 0.7996**2)**0.5, 0.], [0., 1., 0.], [-1., 0., 0.],
    ])
    assert db.search([2., 0., 0.], k=20) == [('same', 100.), ('angled', 80.), ('orthogonal', 0.)]
    assert db.search([1., 0., 0.], k=4, similarity=80) == [('same', 100.), ('angled', 80.)]
    assert db.search([1., 0., 0.], k=1) == [('same', 100.)]
    assert db['same'] == [1., 0., 0.]
    assert db.search([]) == []


@pytest.mark.parametrize('feature', [[1., 0.], [0., 0., 0.], [float('nan'), 0., 1.], [float('inf'), 0., 1.]])
def test_invalid_vectors_do_not_write(db, feature):
    with pytest.raises(ValueError):
        db.add_items(['valid', 'bad'], [[1., 0., 0.], feature])
    assert db.count == 0 and db.next_id == 1
    with pytest.raises(ValueError):
        db.search(feature)


def test_validation(db):
    with pytest.raises(ValueError):
        db.add_items(['a'], [])
    with pytest.raises(ValueError):
        db.add_items([''], [[1., 0., 0.]])
    for similarity in [-1., 101., float('nan')]:
        with pytest.raises(ValueError):
            db.search([1., 0., 0.], similarity=similarity)
    for k in [0, -1, 1.5]:
        with pytest.raises(ValueError):
            db.search([1., 0., 0.], k=k)
    with pytest.raises(ValueError):
        VectorDB('invalid', db.base, 0)
    for name in ['../outside', '.', '..', '/tmp/unsafe', '']:
        with pytest.raises(ValueError):
            VectorDB(name, db.base)
        with pytest.raises(ValueError):
            VectorDB.drop(name, db.base)
    outside = db.base.parent / f'{db.base.name}-outside'
    outside.mkdir()
    (db.base / 'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='inside base_dir'):
        VectorDB('link', db.base)


def test_list_drop_and_legacy_detection(tmp_path):
    assert VectorDB.db_list(tmp_path / 'missing') == []
    VectorDB('b', tmp_path)
    VectorDB('a', tmp_path)
    (tmp_path / 'unrelated').mkdir()
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    (legacy / 'index.db').write_bytes(b'old')
    with pytest.raises(ValueError, match='Legacy HNSW'):
        VectorDB('legacy', tmp_path)
    assert not (legacy / 'images.lance').exists()
    assert VectorDB.db_list(tmp_path) == ['a', 'b']
    assert VectorDB.drop('a', tmp_path)
    assert not VectorDB.drop('missing', tmp_path)
    assert VectorDB.db_list(tmp_path) == ['b']
    with patch('imgsearch.storage.shutil.rmtree', side_effect=OSError):
        assert not VectorDB.drop('b', tmp_path)


def test_index_mutations_and_maintenance(db):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(256, 3)).tolist()
    db.add_items([str(i) for i in range(256)], vectors)
    db.table.create_index('vector', config=IvfHnswFlat(distance_type='cosine', num_partitions=1))
    db.add_items(['new'], [[1., 0., 0.]])
    assert db.search([1., 0., 0.], k=1) == [('new', 100.)]
    db.add_items(['new'], [[0., 1., 0.]])
    assert db.search([0., 1., 0.], k=1) == [('new', 100.)]
    db.delete('new')
    assert 'new' not in [label for label, _ in db.search([0., 1., 0.], k=257)]
    db._changes = 19
    db.add_items(['after'], [[1., 0., 0.]])
    assert db._changes == 0
    assert db.search([1., 0., 0.], k=1) == [('after', 100.)]
    db.clear()
    assert db.search([1., 0., 0.]) == []
    db.add_items(['again'], [[1., 0., 0.]])
    assert db.search([1., 0., 0.]) == [('again', 100.)]


def test_automatic_index(db, monkeypatch):
    monkeypatch.setattr(cfg, 'INDEX_MIN_ROWS', 256)
    rng = np.random.default_rng(7)
    db.add_items([str(i) for i in range(256)], rng.normal(size=(256, 3)).tolist())
    assert any(index.columns == ['vector'] for index in db.table.list_indices())


def test_ann_recall_against_exact_cosine(tmp_path):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(1500, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(32, 32)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    db = VectorDB('ann', tmp_path, dim=32)
    db.add_items([str(i) for i in range(len(vectors))], vectors.tolist())
    db.table.create_index('vector', config=IvfHnswFlat(
        distance_type='cosine', num_partitions=1, m=8, ef_construction=64,
    ))
    recalls = []
    for query in queries:
        expected = set(np.argpartition(-(vectors @ query), 9)[:10])
        found = {int(label) for label, _ in db.search(query.tolist(), k=10)}
        recalls.append(len(expected & found) / 10)
    assert np.mean(recalls) >= 0.97
    assert len(db.search(queries[0].tolist(), k=500)) == 500


def test_maintenance_failure_keeps_committed_write(db, caplog):
    db._changes = 19
    with patch.object(db.table, 'optimize', side_effect=RuntimeError('maintenance')):
        assert db.add_items(['a'], [[1., 0., 0.]])
    assert 'maintenance failed' in caplog.text
    assert db.search([1., 0., 0.]) == [('a', 100.)]


def test_same_instance_concurrent_reads_writes(db):
    def operation(i):
        db.add_items([f'label-{i}'], [[1., 0., 0.]])
        assert db.search([1., 0., 0.])
        return db[f'label-{i}']
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(operation, range(12))) == [[1., 0., 0.]] * 12
    assert db.count == 12
    assert len(set(row['id'] for row in db.table.search().limit(None).to_list())) == 12
