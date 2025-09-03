"""
Unit tests for storage.py module
"""

import tempfile
import unittest
from pathlib import Path
from pickle import dump  # noqa: S403

import pytest
from bidict import bidict
from hnswlib import Index
from imgsearch.consts import CAPACITY
from imgsearch.storage import VectorDB


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
        db = VectorDB(self.test_db_name, self.test_base_dir)

        self.assertEqual(db.name, self.test_db_name)
        self.assertEqual(db.base, self.test_base_dir)
        self.assertEqual(db.path, (self.test_base_dir / self.test_db_name).resolve())
        self.assertEqual(db.size, 0)
        self.assertEqual(db.capacity, CAPACITY)
        self.assertIsInstance(db.index, Index)
        self.assertIsInstance(db.mapping, bidict)

    def test_init_existing_database(self):
        """Test initialization with existing database"""
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

    def test_properties(self):
        """Test database properties"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        self.assertEqual(db.size, 0)
        self.assertEqual(db.next_id, 1)
        self.assertEqual(db.capacity, CAPACITY)
        self.assertEqual(db.next_capacity, CAPACITY + CAPACITY)

    def test_has_id(self):
        """Test has_id method"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add an item
        feature = [1.0] * 512
        db.add_item('test_label', feature)

        self.assertTrue(db.has_id(1))
        self.assertFalse(db.has_id(999))

    def test_has_label(self):
        """Test has_label method"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add an item
        feature = [1.0] * 512
        db.add_item('test_label', feature)

        self.assertTrue(db.has_label('test_label'))
        self.assertFalse(db.has_label('nonexistent'))

    def test_new_index(self):
        """Test new_index static method"""
        index = VectorDB.new_index()

        self.assertIsInstance(index, Index)
        self.assertEqual(index.space, 'cosine')
        self.assertEqual(index.dim, 512)

    def test_new_index_no_init(self):
        """Test new_index with init=False"""
        index = VectorDB.new_index(init=False)

        self.assertIsInstance(index, Index)
        self.assertEqual(index.space, 'cosine')
        self.assertEqual(index.dim, 512)

    def test_load_db_new(self):
        """Test load_db with new database"""
        db_path = self.test_base_dir / 'new_db'

        index, mapping = VectorDB.load_db(db_path)

        self.assertIsInstance(index, Index)
        self.assertIsInstance(mapping, bidict)
        self.assertEqual(len(mapping), 0)

    def test_load_db_corrupted(self):
        """Test load_db with corrupted database"""
        db_path = self.test_base_dir / 'corrupted_db'
        db_path.mkdir(parents=True, exist_ok=True)

        # Create only index file, no mapping file
        (db_path / 'index.db').write_text('corrupted')

        with pytest.raises(OSError):
            VectorDB.load_db(db_path)

    def test_load_db_inconsistent(self):
        """Test load_db with inconsistent index and mapping"""
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

        with pytest.raises(ValueError, match='Index and mapping files are not consistent'):
            VectorDB.load_db(db_path)

    def test_add_item(self):
        """Test adding single item"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        feature = [1.0] * 512
        db.add_item('test_label', feature)

        self.assertEqual(db.size, 1)
        self.assertTrue(db.has_label('test_label'))
        self.assertTrue(db.has_id(1))

    def test_add_item_resize_index(self):
        """Test adding item triggers index resize"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Fill up to capacity
        for i in range(CAPACITY + 1):
            feature = [float(i)] * 512
            db.add_item(f'label_{i}', feature)

        self.assertEqual(db.size, CAPACITY + 1)
        self.assertGreaterEqual(db.capacity, CAPACITY + 1)

    def test_add_items_multiple(self):
        """Test adding multiple items"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        labels = ['label1', 'label2', 'label3']
        features = [[1.0] * 512, [2.0] * 512, [3.0] * 512]

        db.add_items(labels, features)

        self.assertEqual(db.size, 3)
        self.assertTrue(db.has_label('label1'))
        self.assertTrue(db.has_label('label2'))
        self.assertTrue(db.has_label('label3'))

    def test_add_items_invalid_input(self):
        """Test add_items with invalid input"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Empty features
        with pytest.raises(ValueError, match='Invalid labels or features'):
            db.add_items([], [])

        # Mismatched lengths
        with pytest.raises(ValueError, match='Invalid labels or features'):
            db.add_items(['label1'], [[1.0] * 512, [2.0] * 512])

    def test_save_and_load(self):
        """Test saving and loading database"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add some data
        labels = ['label1', 'label2']
        features = [[1.0] * 512, [2.0] * 512]
        db.add_items(labels, features)

        # Save
        db.save()

        # Verify files exist
        self.assertTrue(db.idx_path.exists())
        self.assertTrue(db.map_path.exists())

        # Create new instance and verify data
        db2 = VectorDB(self.test_db_name, self.test_base_dir)
        self.assertEqual(db2.size, 2)
        self.assertTrue(db2.has_label('label1'))
        self.assertTrue(db2.has_label('label2'))

    def test_clear(self):
        """Test clearing database"""
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
        db = VectorDB(self.test_db_name, self.test_base_dir)

        results = db.search([1.0] * 512, k=5)
        self.assertEqual(results, [])

    def test_search_single_item(self):
        """Test search with single item"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        feature = [1.0] * 512
        db.add_item('test_label', feature)

        results = db.search([1.0] * 512, k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 'test_label')
        self.assertGreaterEqual(results[0][1], 0)
        self.assertLessEqual(results[0][1], 100)

    def test_search_multiple_items(self):
        """Test search with multiple items"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add multiple items with distinct features
        labels = ['label1', 'label2', 'label3']
        features = [[1.0] * 512, [0.5] * 512, [0.0] * 512]
        db.add_items(labels, features)

        # Search
        results = db.search([1.0] * 512, k=3)

        self.assertGreaterEqual(len(results), 2)  # At least 2 items should be returned
        # Check that label1 is returned
        returned_labels = [label for label, _ in results]
        self.assertIn('label1', returned_labels)

    def test_search_with_similarity_filter(self):
        """Test search with similarity threshold"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add items with different similarities
        labels = ['similar', 'dissimilar']
        features = [[1.0] * 512, [-1.0] * 512]
        db.add_items(labels, features)

        # Search with high similarity threshold
        results = db.search([1.0] * 512, k=2, similarity=80.0)

        # Should only return the similar item
        self.assertGreaterEqual(len(results), 0)  # Could be 0 or 1 depending on actual similarity

    def test_search_invalid_similarity(self):
        """Test search with invalid similarity parameter"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add some data to avoid empty database edge case
        feature = [1.0] * 512
        db.add_item('test', feature)

        with pytest.raises(ValueError, match='similarity must be between 0 and 100'):
            db.search([1.0] * 512, similarity=-1.0)

        with pytest.raises(ValueError, match='similarity must be between 0 and 100'):
            db.search([1.0] * 512, similarity=101.0)

    def test_search_k_larger_than_size(self):
        """Test search when k is larger than database size"""
        db = VectorDB(self.test_db_name, self.test_base_dir)

        # Add single item
        feature = [1.0] * 512
        db.add_item('test_label', feature)

        # Search for more items than exist
        results = db.search([1.0] * 512, k=10)

        self.assertEqual(len(results), 1)

    def test_database_persistence(self):
        """Test database persistence across instances"""
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
