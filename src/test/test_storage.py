"""
Unit tests for storage.py module
"""

import sys
import tempfile
import unittest
from pathlib import Path
from pickle import dump
from unittest.mock import patch

from bidict import bidict
from hnswlib import Index


class TestVectorDB(unittest.TestCase):
    """Test class for VectorDB"""

    def setUp(self):
        """Setup test environment for each test method"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_base_dir = Path(self.temp_dir.name)
        self.test_db_name = 'test_db'

    def tearDown(self):
        """Cleanup test environment after each test method"""
        self.temp_dir.cleanup()

    def test_init_new_database(self):
        """Test initialization with new database"""
        from imgsearch.config import CAPACITY
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        self.assertEqual(db.name, self.test_db_name)
        self.assertEqual(db.base, self.test_base_dir)
        self.assertEqual(db.path, (self.test_base_dir / self.test_db_name).resolve())
        self.assertEqual(db.size, 0)
        self.assertEqual(db.capacity, CAPACITY)
        self.assertEqual(db.next_id, 1)
        self.assertEqual(db.next_capacity, CAPACITY + CAPACITY)
        self.assertIsInstance(db.index, Index)
        self.assertIsInstance(db.mapping, bidict)
        self.assertEqual(len(db.mapping), 0)

    def test_init_existing_database(self):
        """Test initialization with existing database"""
        from imgsearch.storage import VectorDB

        # First create a database
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add some data
        feature = [1.0] * 512
        db.add_item('test_label', feature)
        db.save()

        # Create new instance with same path
        db2 = VectorDB(self.test_db_name, self.test_base_dir)

        self.assertEqual(db2.size, 1)
        self.assertIn('test_label', db2.mapping.inv)
        self.assertEqual(db2.next_id, 2)

    def test_init_default_database(self):
        """Test initialization with default database parameters"""
        # Clear sys.modules to avoid importing existing modules
        for name in sys.modules.copy():
            if name.startswith('imgsearch'):
                del sys.modules[name]

        with patch('imgsearch.config.BASE_DIR', self.test_base_dir):
            from imgsearch.config import CAPACITY, DB_NAME
            from imgsearch.storage import VectorDB

            db = VectorDB()

            self.assertEqual(db.name, DB_NAME)
            self.assertEqual(db.base, self.test_base_dir)
            self.assertEqual(db.path, (self.test_base_dir / DB_NAME).resolve())
            self.assertEqual(db.size, 0)
            self.assertEqual(db.capacity, CAPACITY)

    def test_has_id(self):
        """Test has_id method"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add an item
        feature = [1.0] * 512
        db.add_item('test_label', feature)

        self.assertTrue(db.has_id(1))
        self.assertFalse(db.has_id(999))

    def test_has_label(self):
        """Test has_label method"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add an item
        feature = [1.0] * 512
        db.add_item('test_label', feature)

        self.assertTrue(db.has_label('test_label'))
        self.assertFalse(db.has_label('nonexistent'))

    def test_new_index(self):
        """Test new_index static method"""
        from imgsearch.config import CAPACITY
        from imgsearch.storage import VectorDB

        index = VectorDB.new_index()

        self.assertIsInstance(index, Index)
        self.assertEqual(index.space, 'cosine')
        self.assertEqual(index.dim, 512)
        self.assertEqual(index.max_elements, CAPACITY)
        self.assertEqual(index.ef_construction, 400)
        self.assertEqual(index.M, 32)

    def test_new_index_no_init(self):
        """Test new_index with init=False"""
        from imgsearch.storage import VectorDB

        index = VectorDB.new_index(init=False)

        self.assertIsInstance(index, Index)
        self.assertEqual(index.space, 'cosine')
        self.assertEqual(index.dim, 512)

    def test_load_db_new(self):
        """Test load_db with new database"""
        from imgsearch.storage import VectorDB

        db_path = self.test_base_dir / 'new_db'

        index, mapping = VectorDB.load_db(db_path)

        self.assertIsInstance(index, Index)
        self.assertIsInstance(mapping, bidict)
        self.assertEqual(len(mapping), 0)

    def test_load_db_corrupted(self):
        """Test load_db with corrupted database"""
        from imgsearch.storage import VectorDB

        db_path = self.test_base_dir / 'corrupted_db'
        db_path.mkdir(parents=True, exist_ok=True)

        # Create only index file, no mapping file
        (db_path / 'index.db').write_text('corrupted')

        with self.assertRaises(OSError):
            VectorDB.load_db(db_path)

    def test_load_db_inconsistent(self):
        """Test load_db with inconsistent index and mapping"""
        from imgsearch.storage import VectorDB

        db_path = self.test_base_dir / 'inconsistent_db'
        db_path.mkdir(parents=True, exist_ok=True)

        # Create a valid index
        index = VectorDB.new_index(init=False)
        index.init_index(max_elements=10, ef_construction=200, M=16)
        index.add_items([[1.0] * 512], [1])
        index.save_index(str(db_path / 'index.db'))

        # Create inconsistent mapping
        with open(db_path / 'mapping.db', 'wb') as f:
            dump(bidict({}), f)  # Empty mapping but index has 1 item

        with self.assertRaises(ValueError) as cm:
            VectorDB.load_db(db_path)
        self.assertEqual(str(cm.exception), 'Index and mapping files are not consistent')

    def test_add_item(self):
        """Test adding single item"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        feature = [1.0] * 512
        db.add_item('test_label', feature)

        self.assertEqual(db.size, 1)
        self.assertTrue(db.has_label('test_label'))
        self.assertTrue(db.has_id(1))
        self.assertEqual(db.next_id, 2)
        self.assertIn(1, db.mapping)
        self.assertEqual(db.mapping[1], 'test_label')

    def test_add_item_resize_index(self):
        """Test adding item triggers index resize"""
        from imgsearch.config import CAPACITY
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Fill up to capacity - 1, then add one more to trigger resize
        for i in range(CAPACITY):
            feature = [float(i)] * 512
            db.add_item(f'label_{i}', feature)

        # Add one more to trigger resize
        feature = [float(CAPACITY)] * 512
        db.add_item(f'label_{CAPACITY}', feature)

        self.assertEqual(db.size, CAPACITY + 1)
        self.assertGreaterEqual(db.capacity, CAPACITY + CAPACITY)
        self.assertTrue(db.has_label(f'label_{CAPACITY}'))

    def test_add_items_multiple(self):
        """Test adding multiple items"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        labels = ['label1', 'label2', 'label3']
        features = [[1.0] * 512, [2.0] * 512, [3.0] * 512]

        db.add_items(labels, features)

        self.assertEqual(db.size, 3)
        self.assertTrue(db.has_label('label1'))
        self.assertTrue(db.has_label('label2'))
        self.assertTrue(db.has_label('label3'))
        self.assertEqual(db.next_id, 4)
        self.assertEqual(list(db.mapping.keys()), [1, 2, 3])
        self.assertEqual(db.mapping[1], 'label1')

    def test_add_items_invalid_input(self):
        """Test add_items with invalid input"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Empty features
        with self.assertRaises(ValueError, msg='Invalid labels or features'):
            db.add_items([], [])

        # Mismatched lengths
        with self.assertRaises(ValueError, msg='Invalid labels or features'):
            db.add_items(['label1'], [[1.0] * 512, [2.0] * 512])

    def test_save_and_load(self):
        """Test saving and loading database"""
        from imgsearch.config import IDX_NAME, MAP_NAME
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add some data
        labels = ['label1', 'label2']
        features = [[1.0] * 512, [2.0] * 512]
        db.add_items(labels, features)

        # Save
        db.save()

        # Verify files exist with correct names
        self.assertTrue((self.test_base_dir / self.test_db_name / IDX_NAME).exists())
        self.assertTrue((self.test_base_dir / self.test_db_name / MAP_NAME).exists())

        # Create new instance and verify data persistence
        db2 = VectorDB(self.test_db_name, self.test_base_dir)
        self.assertEqual(db2.size, 2)
        self.assertTrue(db2.has_label('label1'))
        self.assertTrue(db2.has_label('label2'))
        self.assertEqual(db2.mapping[1], 'label1')
        self.assertEqual(db2.mapping[2], 'label2')

    def test_clear(self):
        """Test clearing database"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add some data
        feature = [1.0] * 512
        db.add_item('test_label', feature)
        self.assertEqual(db.size, 1)

        # Clear
        db.clear()

        self.assertEqual(db.size, 0)
        self.assertEqual(len(db.mapping), 0)

    def test_search_empty_database(self):
        """Test search on empty database"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        results = db.search([1.0] * 512, k=5)
        self.assertEqual(results, [])

        # Test with similarity filter
        results = db.search([1.0] * 512, k=5, similarity=50.0)
        self.assertEqual(results, [])

    def test_search_single_item(self):
        """Test search with single item"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        feature = [1.0] * 512
        db.add_item('test_label', feature)

        results = db.search([1.0] * 512, k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 'test_label')
        self.assertIsInstance(results[0][1], float)
        self.assertGreaterEqual(results[0][1], 0.0)
        self.assertLessEqual(results[0][1], 100.0)

    def test_search_multiple_items(self):
        """Test search with multiple items"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add multiple items with distinct features
        labels = ['label1', 'label2', 'label3']
        features = [[1.0] * 512, [0.5] * 512, [0.0] * 512]
        db.add_items(labels, features)

        # Search
        results = db.search([1.0] * 512, k=3)

        self.assertEqual(len(results), 3)
        # Check that all labels are returned and similarities are reasonable
        returned_labels = [label for label, _ in results]
        self.assertCountEqual(returned_labels, labels)
        similarities = [sim for _, sim in results]
        self.assertTrue(all(0 <= s <= 100 for s in similarities))

    def test_search_with_similarity_filter(self):
        """Test search with similarity threshold"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add items with different similarities to query [1.0]*512
        similar_feature = [1.0] * 512  # similarity = 100%
        dissimilar_feature = [-1.0] * 512  # similarity = 0%
        labels = ['similar', 'dissimilar']
        features = [similar_feature, dissimilar_feature]
        db.add_items(labels, features)

        # Search with high similarity threshold
        results = db.search([1.0] * 512, k=2, similarity=80.0)

        # Should only return the similar item
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 'similar')
        self.assertGreaterEqual(results[0][1], 80.0)

    def test_search_invalid_similarity(self):
        """Test search with invalid similarity parameter"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add some data to avoid empty database edge case
        feature = [1.0] * 512
        db.add_item('test', feature)

        with self.assertRaises(ValueError, msg='similarity must be between 0 and 100'):
            db.search([1.0] * 512, similarity=-1.0)

        with self.assertRaises(ValueError, msg='similarity must be between 0 and 100'):
            db.search([1.0] * 512, similarity=101.0)

    def test_search_k_larger_than_size(self):
        """Test search when k is larger than database size"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add single item
        feature = [1.0] * 512
        db.add_item('test_label', feature)

        # Search for more items than exist
        results = db.search([1.0] * 512, k=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 'test_label')

    def test_database_persistence(self):
        """Test database persistence across instances"""
        from imgsearch.storage import VectorDB

        # Create first instance
        db1 = VectorDB(self.test_db_name, self.test_base_dir)

        # Add data
        labels = ['persistent1', 'persistent2']
        features = [[1.0] * 512, [2.0] * 512]
        db1.add_items(labels, features)
        db1.save()

        # Create second instance
        db2 = VectorDB(self.test_db_name, self.test_base_dir)

        # Verify data persistence
        self.assertEqual(db2.size, 2)
        self.assertTrue(db2.has_label('persistent1'))
        self.assertTrue(db2.has_label('persistent2'))
        self.assertEqual(db2.next_id, 3)

    def test_clear_database(self):
        """Test clearing database"""
        from imgsearch.config import CAPACITY
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add some data
        feature = [1.0] * 512
        db.add_item('test_label', feature)
        self.assertEqual(db.size, 1)

        # Clear
        db.clear()

        self.assertEqual(db.size, 0)
        self.assertEqual(len(db.mapping), 0)
        self.assertEqual(db.next_id, 1)
        self.assertEqual(db.capacity, CAPACITY)

    def test_db_list(self):
        """Test listing available databases"""
        from imgsearch.storage import VectorDB

        # Create two test databases
        db1 = VectorDB('db1', self.test_base_dir)
        feature1 = [1.0] * 512
        db1.add_item('item1', feature1)
        db1.save()

        db2 = VectorDB('db2', self.test_base_dir)
        feature2 = [2.0] * 512
        db2.add_item('item2', feature2)
        db2.save()

        # List databases
        databases = db1.db_list()

        self.assertEqual(len(databases), 2)
        self.assertIn('db1', databases)
        self.assertIn('db2', databases)

    def test_has_labels_multiple(self):
        """Test has_labels method with multiple labels"""
        from imgsearch.storage import VectorDB

        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add some labels
        labels = ['label1', 'label2', 'label3']
        features = [[1.0] * 512, [2.0] * 512, [3.0] * 512]
        db.add_items(labels, features)

        # Test has_labels
        has_results = db.has_labels('label1', 'label2', 'nonexistent', 'label3')

        self.assertEqual(len(has_results), 4)
        self.assertEqual(has_results, [True, True, False, True])
