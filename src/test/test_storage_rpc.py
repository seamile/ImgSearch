"""Exercise the updated client/RPC boundary without a daemon or model download."""

import logging
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, PropertyMock, patch

import pytest

from imgsearch.client import Client, create_parser
from imgsearch.server import RPCService


@pytest.fixture
def service(tmp_path):
    # Avoid starting the perpetual ingestion worker for these storage-only tests.
    with patch('threading.Thread.start'):
        service = RPCService(tmp_path, logger=logging.getLogger('storage-test'))
    service.databases.clear()
    yield service
    service.databases.clear()


def test_client_rpc_delete_info_unload(service):
    db = service._get_db('test')
    db.add_items(['a'], [[1.] + [0.] * 511])
    client = Client('test')
    with patch.object(Client, 'service', new_callable=PropertyMock, return_value=service):
        info = client.get_db_info()
        assert info['count'] == 1 and info['size'] > 0
        assert 'capacity' not in info
        assert client.delete_images(['a'])
        assert not client.delete_images(['missing'])
        assert client.get_db_info()['count'] == 0
        assert client.unload_dbs('test') == 1
    assert service._get_db('test').count == 0
    assert service.handle_clear_db('test')
    assert service.handle_drop_db('test')
    assert service.handle_get_db_info('test') is None


def test_concurrent_database_initialization(service):
    with ThreadPoolExecutor(max_workers=4) as pool:
        databases = list(pool.map(lambda _: service._get_db('test'), range(8)))
    assert all(db is databases[0] for db in databases)


def test_ingestion_and_text_search_use_lance_storage(service):
    vector = [1.] + [0.] * 511
    service.clip = Mock()
    service.clip.embed_images.return_value = [vector]
    service.clip.embed_text.return_value = vector
    service._process_images([], ['image'], 'test')
    assert service.handle_search('query', db_name='test') == [('image', 100.)]
    assert service.handle_check_exist_labels(['image', 'missing'], 'test') == [True, False]


def test_removed_rebuild_cli():
    parser = create_parser()
    args = parser.parse_args(['db', 'test', '--delete', 'a'])
    assert args.delete == ['a'] and not hasattr(args, 'rebuild')
    with pytest.raises(SystemExit):
        parser.parse_args(['db', 'test', '--rebuild', '--delete', 'a'])
