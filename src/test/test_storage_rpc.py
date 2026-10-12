"""RPC and CLI behavior without model downloads or a running daemon."""

import logging
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
from PIL import Image
from Pyro5.serializers import serializers

from imgsearch.client import Client, create_parser, main
from imgsearch.server import RPCService
from imgsearch.utils import bytes2img, img2bytes


@pytest.fixture
def service(tmp_path):
    with patch('threading.Thread.start'):
        return RPCService(tmp_path, logger=logging.getLogger('storage-test'))


def feature(i=0):
    value = np.zeros(512, dtype=np.float32)
    value[i] = 1
    return value


def test_client_rpc_delete_info_unload(service):
    db = service._get_db('test')
    row = db.model.from_feature(feature(), label='a')
    db.add_items([row])
    client = Client('test')
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service):
        assert client.get_db_info()['count'] == 1
        assert client.get_images(['a'])[0]['id'] == row.id
        assert client.delete_images([row.id], by_id=True)
        assert not client.delete_images(['missing'])
        assert client.unload_dbs('test') == 1
    assert service._get_db('test').count == 0
    assert service.handle_clear_db('test')
    assert service.handle_drop_db('test')
    assert service.handle_get_db_info('test') is None


def test_concurrent_initialization_and_separate_roots(service, tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        tables = list(pool.map(lambda _: service._get_db('test'), range(8)))
    assert all(table is tables[0] for table in tables)
    with patch('threading.Thread.start'):
        other = RPCService(tmp_path / 'other')
    assert other._get_db('test').path != tables[0].path


def test_ingestion_update_and_search_wire_payload(service, tmp_path):
    service.clip = Mock()
    service.clip.embed_images.return_value = [feature()]
    service.clip.embed_text.return_value = feature()
    for name, update in [('a', False), ('b', False), ('c', True)]:
        service._process_images([Image.new('RGB', (1, 1))], [str(tmp_path / f'{name}.jpg')], 'test', update)
    hits = service.handle_search('query', db_name='test')
    assert hits[0]['label'] == 'c' and len(hits[0]['paths']) == 3
    assert set(hits[0]) == {'id', 'label', 'paths', 'score'}
    assert serializers['msgpack'].loads(serializers['msgpack'].dumps(hits)) == hits


def test_client_add_same_names_and_flag(service, tmp_path):
    paths = []
    for folder in ['one', 'two']:
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / 'same.png'
        Image.new('RGB', (2, 2)).save(path)
        paths.append(str(path))
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service):
        assert Client('test').add_images(paths, update_label=True) == 2
    queued = [service.image_queue.get_nowait() for _ in range(2)]
    assert {entry[0] for entry in queued} == set(paths)
    assert all(entry[3] for entry in queued)


def test_delete_preview_freezes_ids(service, monkeypatch, capsys, tmp_path):
    db = service._get_db('test')
    rows = [db.model.from_feature(feature(i), label='same', paths=[str(tmp_path / f'{i}.jpg')]) for i in range(3)]
    db.add_items(rows[:2])

    def confirm(prompt):
        output = capsys.readouterr().out
        assert all(row.id in output and row.paths[0] in output for row in rows[:2])
        db.add_items([rows[2]])
        return 'y'
    monkeypatch.setattr('builtins.input', confirm)
    monkeypatch.setattr('sys.argv', ['isearch', 'db', 'test', '--delete', 'same'])
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service):
        main()
    assert db.count == 1 and rows[2].id in db


def test_delete_id_cancel_and_missing_label(service, monkeypatch):
    db = service._get_db('test')
    row = db.model.from_feature(feature(), label='a')
    db.add_items([row])
    monkeypatch.setattr('builtins.input', lambda _: 'n')
    monkeypatch.setattr('sys.argv', ['isearch', 'db', 'test', '--delete-id', row.id])
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service):
        main()
        monkeypatch.setattr('sys.argv', ['isearch', 'db', 'test', '--delete', 'a', 'missing'])
        with pytest.raises(SystemExit):
            main()
    assert db.count == 1


def test_prune_missing_and_duplicate_paths_without_inference(service, tmp_path):
    db = service._get_db('default')
    good = tmp_path / 'good.png'
    good.write_bytes(b'present')
    missing = tmp_path / 'missing.png'
    row = db.model.from_feature(feature(), label='a', paths=[str(good), str(missing)])
    db.add_items([row])
    db.table.update(where=f"id = '{row.id}'", values={'paths': [str(good), str(good), str(missing)]})
    service.clip = Mock()
    stats = service.handle_prune_db()
    assert stats['deduplicated_paths'] == stats['removed_paths'] == 1
    assert db[row.id].paths == [str(good)]
    good.unlink()
    service.handle_prune_db()
    assert db[row.id].paths == [] and db.count == 1
    service.clip.embed_image.assert_not_called()


@pytest.mark.parametrize('target', [0, 2, 3])
def test_prune_cross_id_matching_existing_and_new(service, tmp_path, target):
    db = service._get_db('default')
    p, q, r = [tmp_path / f'{name}.png' for name in ['p', 'q', 'r']]
    for path in [p, q, r]:
        Image.new('RGB', (8, 5), color='red').save(path)
    rows = [db.model.from_feature(feature(i), label=f'label-{i}', paths=paths)
            for i, paths in enumerate([[str(p), str(q)], [str(p), str(r)], []])]
    db.add_items(rows)
    service.clip = Mock()
    service.clip.embed_image.return_value = feature(target)
    stats = service.handle_prune_db()
    expected = db.model.from_path(feature(target), p)
    assert db[expected.id].label == (f'label-{target}' if target < 3 else 'p')
    assert str(p) in db[expected.id].paths
    assert str(q) in db[rows[0].id].paths and str(r) in db[rows[1].id].paths
    for row in rows[:2]:
        assert (str(p) in db[row.id].paths) == (row.id == expected.id)
    assert stats['recomputed_paths'] == 1
    assert stats['inserted_rows'] == int(target == 3)
    with Image.open(p) as image:
        expected_pixels = bytes2img(img2bytes(image, 384)).tobytes()
    assert service.clip.embed_image.call_args.args[0].tobytes() == expected_pixels


def test_prune_multiple_conflicts_and_failure(service, tmp_path):
    db = service._get_db('default')
    paths = [tmp_path / f'{name}.png' for name in ['p', 'q']]
    for path in paths:
        Image.new('RGB', (2, 2)).save(path)
    rows = [db.model.from_feature(feature(i), label=str(i), paths=list(map(str, paths))) for i in range(2)]
    db.add_items(rows)
    service.clip = Mock()
    service.clip.embed_image.side_effect = RuntimeError('inference failed')
    stats = service.handle_prune_db()
    assert stats['errors'] == 2 and all(len(db[row.id].paths) == 2 for row in rows)
    service.clip.embed_image.side_effect = None
    service.clip.embed_image.return_value = feature(2)
    stats = service.handle_prune_db()
    target = db.model.from_path(feature(2), paths[0])
    assert db[target.id].paths == list(map(str, paths))
    assert all(db[row.id].paths == [] for row in rows)
    assert db.count == 3 and stats['inserted_rows'] == 1


def test_prune_permission_error_preserves_path(service, tmp_path):
    db = service._get_db('default')
    path = tmp_path / 'private.png'
    row = db.model.from_path(feature(), path)
    db.add_items([row])
    original = type(path).stat

    def stat(target, *args, **kwargs):
        if target == path:
            raise PermissionError('denied')
        return original(target, *args, **kwargs)
    with patch('pathlib.Path.stat', stat):
        assert service.handle_prune_db()['errors'] == 1
    assert db[row.id].paths == [str(path)]


def test_cli_prune_defaults_and_parser(service, monkeypatch):
    service._get_db('default')
    service._get_db('other')
    parser = create_parser()
    assert parser.parse_args(['db', '--prune', 'other']).db_name == 'other'
    assert parser.parse_args(['add', '--update-label', '/images/a.jpg']).update_label
    with pytest.raises(SystemExit):
        parser.parse_args(['add', '-l', 'name', '/images/a.jpg'])
    monkeypatch.setattr('sys.argv', ['isearch', 'db', '--prune'])
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service), patch.object(service, 'handle_prune_db', wraps=service.handle_prune_db) as prune:
        main()
        prune.assert_called_once_with('default')


def test_search_opens_first_accessible_path(service, tmp_path, monkeypatch, capsys):
    db = service._get_db('default')
    existing = tmp_path / 'existing.png'
    existing.write_bytes(b'present')
    db.add_items([
        db.model.from_feature(feature(), label='with-path', paths=[str(tmp_path / 'missing'), str(existing)]),
        db.model.from_feature(feature(1), label='without-path'),
    ])
    service.clip = Mock()
    service.clip.embed_text.return_value = feature()
    monkeypatch.setattr('sys.argv', ['isearch', 'search', 'query', '-o'])
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service), patch('imgsearch.utils.open_images') as opened:
        main()
    opened.assert_called_once_with([str(existing)])
    assert 'No accessible path' in capsys.readouterr().err
