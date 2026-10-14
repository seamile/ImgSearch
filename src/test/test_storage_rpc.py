"""RPC and CLI behavior without model downloads or a running daemon."""

import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
from PIL import Image
from Pyro5.serializers import serializers

from imgsearch import config as cfg
from imgsearch.client import Client, create_parser, main
from imgsearch.server import RPCService
from imgsearch.utils import bytes2img, get_logger, img2bytes


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


def test_maintenance_uses_service_log_file(service, tmp_path):
    with patch('sys.stderr.isatty', return_value=False):
        logger = get_logger('storage-file-test', log_dir=tmp_path)
    service.logger = logger
    try:
        db = service._get_db('test')
        assert db.logger is logger
        db._changes = 19
        db._maintain()
        output = (tmp_path / 'isearch.log').read_text()
        assert '[Storage] Optimizing table "test"' in output
        assert '[Storage] Optimization completed for "test"' in output
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def test_concurrent_initialization_and_separate_roots(service, tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        tables = list(pool.map(lambda _: service._get_db('test'), range(8)))
    assert all(table is tables[0] for table in tables)
    with patch('threading.Thread.start'):
        other = RPCService(tmp_path / 'other')
    assert other._get_db('test').path != tables[0].path


def test_ingestion_update_and_search_wire_payload(service, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=service.logger.name)
    service.clip = Mock()
    service.clip.embed_images.return_value = [feature()]
    service.clip.embed_text.return_value = feature()
    for name, update in [('a', False), ('b', False), ('c', True)]:
        service._process_images([Image.new('RGB', (1, 1))], [str(tmp_path / f'{name}.jpg')], 'test', update)
    hits = service.handle_search('query', db_name='test')
    assert hits[0]['label'] == 'c' and len(hits[0]['paths']) == 3
    assert set(hits[0]) == {'id', 'label', 'paths', 'score'}
    assert serializers['msgpack'].loads(serializers['msgpack'].dumps(hits)) == hits
    assert '[Add] Computing vectors for 1 images in "test"' in caplog.text
    assert '[Add] Saving 1 image vectors to "test"' in caplog.text
    assert '[Add] Completed 1 images in "test": 1 records changed' in caplog.text
    assert '[Search] Completed in "test": 1 results' in caplog.text


def test_client_add_same_names_and_flag(service, tmp_path, capsys):
    paths = []
    for folder in ['one', 'two']:
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / 'same.png'
        Image.new('RGB', (2, 2)).save(path)
        paths.append(str(path))
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service):
        assert Client('test').add_images(paths, update_label=True, dedup_mode='strict') == 2
    queued = [service.image_queue.get_nowait() for _ in range(2)]
    assert {entry[0] for entry in queued} == set(paths)
    assert all(entry[3] for entry in queued)
    assert all(entry[4] == 'strict' for entry in queued)
    output = capsys.readouterr()
    assert output.out == ''
    assert all(path in output.err for path in paths)
    assert 'Found 2 image files' in output.err
    assert 'Preparing 2 images' in output.err
    assert 'Queued 2 images' in output.err


def test_fast_add_skips_existing_label_before_reading(service, tmp_path, capsys):
    path = tmp_path / "it's here.png"
    Image.new('RGB', (2, 2)).save(path)
    db = service._get_db('test')
    db.add_items([db.model.from_feature(feature(), label=path.stem)])
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service), patch('imgsearch.client.Image.open') as opened:
        assert Client('test').add_images([str(path)]) == 0
    opened.assert_not_called()
    assert service.image_queue.empty()
    output = capsys.readouterr().err
    assert 'Skipped 1 images with existing labels' in output
    assert 'Preparing' not in output and 'Queued' not in output


def test_add_reports_partial_acceptance(service, tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO, logger=service.logger.name)
    image = img2bytes(Image.new('RGB', (2, 2)))
    assert (
        service.handle_add_images({str(tmp_path / 'good.png'): image, str(tmp_path / 'bad.png'): b'bad'}, 'test') == 1
    )
    assert 'Queued 1/2 images for "test"' in caplog.text
    assert service.image_queue.qsize() == 1

    paths = [tmp_path / name for name in ('one.png', 'two.png')]
    for path in paths:
        Image.new('RGB', (2, 2)).save(path)
    with (
        patch.object(Client, 'service', new_callable=PropertyMock, return_value=service),
        patch.object(service, 'handle_add_images', return_value=1),
    ):
        assert Client('test').add_images([str(path) for path in paths], dedup_mode='strict') == 1
    output = capsys.readouterr().err
    assert 'Server accepted 1/2 images; 1 were not queued' in output
    assert 'Queued 2 images' not in output


def test_failed_embedding_is_not_reported_as_completed(service, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=service.logger.name)
    service.clip = Mock()
    service.clip.embed_images.side_effect = RuntimeError('inference failed')
    service._process_images([Image.new('RGB', (2, 2))], [str(tmp_path / 'one.png')], 'test')
    assert 'Failed to process 1 images in "test"' in caplog.text
    assert 'inference failed' in caplog.text
    assert '[Add] Completed' not in caplog.text
    assert service._get_db('test').count == 0


def test_add_empty_directory_reports_no_images(tmp_path, capsys):
    with patch.object(Client, 'service', new_callable=PropertyMock) as remote:
        assert Client('test').add_images([str(tmp_path)]) == 0
    remote.assert_not_called()
    output = capsys.readouterr()
    assert 'No image files found' in output.err and output.out == ''


def test_prune_reports_progress_and_preserves_empty_records(service, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=service.logger.name)
    db = service._get_db('test')
    row = db.model.from_feature(feature(), label='missing', paths=[str(tmp_path / f'{i}.png') for i in range(1001)])
    db.add_items([row])
    stats = service.handle_prune_db('test')
    assert stats['checked_paths'] == stats['removed_paths'] == 1001
    assert db[row.id].paths == []
    assert 'Checked 1000/1001 paths in "test"' in caplog.text
    assert '[Prune] Completed in "test"' in caplog.text


@pytest.mark.parametrize('mode, expected', [('fast', 1), ('strict', 3)])
def test_dedup_before_embedding_including_queued_labels(service, tmp_path, mode, expected):
    db = service._get_db('test')
    original = db.model.from_feature(feature(), label='known')
    db.add_items([original])
    paths = [str(tmp_path / path) for path in ['known.png', 'one/new.png', 'two/new.jpg']]
    images = [Image.new('RGB', (2, 2)) for _ in paths]
    service.clip = Mock()
    service.clip.embed_images.side_effect = lambda batch: [feature(i + 1) for i in range(len(batch))]
    service._process_images(images, paths, 'test', dedup_mode=mode)
    assert len(service.clip.embed_images.call_args.args[0]) == expected
    assert db.count == expected + 1
    assert db[original.id].paths == []
    if mode == 'fast':
        assert db.get_info(['new'])[0]['paths'] == [paths[1]]
        service.clip.embed_images.reset_mock()
        service._process_images(images, paths, 'test', dedup_mode=mode)
        service.clip.embed_images.assert_not_called()


def test_queue_preserves_mixed_dedup_modes(service, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, 'BATCH_SIZE', 2)
    db = service._get_db('test')
    original = db.model.from_feature(feature(), label='known')
    db.add_items([original])
    service.clip = Mock()
    service.clip.embed_images.return_value = [feature()]
    data = img2bytes(Image.new('RGB', (2, 2)))
    assert service.handle_add_images({str(tmp_path / 'known.png'): data}, 'test') == 1
    assert service.handle_add_images(
        {str(tmp_path / 'renamed.png'): data}, 'test', update_label=True, dedup_mode='strict',
    ) == 1
    entries = [service.image_queue.get_nowait() for _ in range(2)]
    with patch.object(service.image_queue, 'get', side_effect=[*entries, Empty(), KeyboardInterrupt()]):
        with pytest.raises(KeyboardInterrupt):
            service._process_queue()
    service.clip.embed_images.assert_called_once()
    assert db.count == 1
    assert db[original.id].label == 'renamed'
    assert db[original.id].paths == [str(tmp_path / 'renamed.png')]


def test_existing_labels_scope_and_missing_keys(service):
    db = service._get_db('test')
    labels = [f'label-{i}' for i in range(12)] + ["it's here"]
    db.add_items([db.model.from_feature(feature(i), label=label) for i, label in enumerate(labels)])
    assert service.handle_existing_labels([*labels, 'missing', labels[0]], 'test') == sorted(labels)
    assert service.handle_existing_labels(labels, 'other') == []
    assert service.handle_existing_labels([], 'test') == []


@pytest.mark.parametrize('mode', ['fast', 'strict'])
def test_add_cli_forwards_dedup_mode(monkeypatch, mode):
    monkeypatch.setattr('sys.argv', ['isearch', 'add', '--dedup-mode', mode, '--update-label', '/images'])
    with patch.object(Client, 'add_images', return_value=0) as add:
        main()
    add.assert_called_once_with(['/images'], update_label=True, dedup_mode=mode)


def test_add_dedup_mode_defaults_and_validation(service):
    parser = create_parser()
    assert parser.parse_args(['add', '/images']).dedup_mode == 'fast'
    with pytest.raises(SystemExit):
        parser.parse_args(['add', '--dedup-mode', 'unknown', '/images'])
    with pytest.raises(ValueError, match='dedup_mode'):
        Client('test').add_images([], dedup_mode='unknown')
    with pytest.raises(ValueError, match='dedup_mode'):
        service.handle_add_images({}, 'test', dedup_mode='unknown')


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
