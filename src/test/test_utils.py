"""
Unit tests for utils.py module
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from imgsearch.utils import (
    ColorFormatter,
    bytes2img,
    find_all_images,
    get_logger,
    ibatch,
    img2bytes,
    is_image,
    open_images,
    print_err,
    print_warn,
)
from PIL import Image


class TestUtils(unittest.TestCase):
    """Test class for utility functions"""

    def setUp(self):
        """Setup test environment for each test method"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Cleanup test environment after each test method"""
        self.temp_dir.cleanup()

    def test_is_image_with_valid_image(self):
        """Test is_image with valid image file"""
        # Create a test image
        img_path = self.test_dir / 'test.jpg'
        img = Image.new('RGB', (10, 10), color='red')
        img.save(img_path)

        self.assertTrue(is_image(img_path))

    def test_is_image_with_non_image_file(self):
        """Test is_image with non-image file"""
        txt_path = self.test_dir / 'test.txt'
        txt_path.write_text('This is not an image')

        self.assertFalse(is_image(txt_path))

    def test_is_image_with_hidden_file(self):
        """Test is_image with hidden file"""
        hidden_path = self.test_dir / '.hidden.jpg'
        img = Image.new('RGB', (10, 10), color='red')
        img.save(hidden_path)

        self.assertFalse(is_image(hidden_path, ignore_hidden=True))
        self.assertTrue(is_image(hidden_path, ignore_hidden=False))

    def test_is_image_with_directory(self):
        """Test is_image with directory path"""
        sub_dir = self.test_dir / 'subdir'
        sub_dir.mkdir()

        self.assertFalse(is_image(sub_dir))

    def test_img2bytes_basic(self):
        """Test img2bytes basic functionality"""
        img = Image.new('RGB', (100, 100), color='blue')
        result = img2bytes(img)

        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_img2bytes_with_resize(self):
        """Test img2bytes with resize parameter"""
        img = Image.new('RGB', (200, 200), color='green')
        result = img2bytes(img, resize=100)

        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_img2bytes_rgba_conversion(self):
        """Test img2bytes converts RGBA to RGB"""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        result = img2bytes(img)

        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_bytes2img(self):
        """Test bytes2img functionality"""
        # Create test image
        original_img = Image.new('RGB', (50, 50), color='yellow')
        img_bytes = img2bytes(original_img)

        # Convert back
        result_img = bytes2img(img_bytes)

        self.assertIsInstance(result_img, Image.Image)
        self.assertEqual(original_img.tobytes(), result_img.tobytes())

    def test_find_all_images_single_file(self):
        """Test find_all_images with single image file"""
        img_path = self.test_dir / 'single.jpg'
        img = Image.new('RGB', (10, 10), color='red')
        img.save(img_path)

        results = list(find_all_images(str(img_path)))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], img_path)

    def test_find_all_images_directory(self):
        """Test find_all_images with directory"""
        # Create test images in directory
        img1 = self.test_dir / 'img1.jpg'
        img2 = self.test_dir / 'img2.png'
        txt_file = self.test_dir / 'readme.txt'

        Image.new('RGB', (10, 10), color='red').save(img1)
        Image.new('RGB', (10, 10), color='blue').save(img2)
        txt_file.write_text('Not an image')

        results = list(find_all_images(str(self.test_dir)))

        self.assertEqual(len(results), 2)
        self.assertIn(img1, results)
        self.assertIn(img2, results)

    def test_find_all_images_recursive(self):
        """Test find_all_images with recursive search"""
        sub_dir = self.test_dir / 'subdir'
        sub_dir.mkdir()

        img1 = self.test_dir / 'root.jpg'
        img2 = sub_dir / 'nested.png'

        Image.new('RGB', (10, 10), color='red').save(img1)
        Image.new('RGB', (10, 10), color='blue').save(img2)

        results = list(find_all_images(str(self.test_dir), recursively=True))

        self.assertEqual(len(results), 2)
        self.assertIn(img1, results)
        self.assertIn(img2, results)

    def test_find_all_images_non_recursive(self):
        """Test find_all_images without recursive search"""
        sub_dir = self.test_dir / 'subdir'
        sub_dir.mkdir()

        img1 = self.test_dir / 'root.jpg'
        img2 = sub_dir / 'nested.png'

        Image.new('RGB', (10, 10), color='red').save(img1)
        Image.new('RGB', (10, 10), color='blue').save(img2)

        results = list(find_all_images(str(self.test_dir), recursively=False))

        self.assertEqual(len(results), 1)
        self.assertIn(img1, results)
        self.assertNotIn(img2, results)

    def test_find_all_images_mixed_paths(self):
        """Test find_all_images with mixed file and directory paths"""
        # Create files
        img_file = self.test_dir / 'direct.jpg'
        sub_dir = self.test_dir / 'subdir'
        sub_dir.mkdir()
        nested_img = sub_dir / 'nested.png'

        Image.new('RGB', (10, 10), color='red').save(img_file)
        Image.new('RGB', (10, 10), color='blue').save(nested_img)

        paths = [str(img_file), str(sub_dir)]
        results = list(find_all_images(paths))

        self.assertEqual(len(results), 2)
        self.assertIn(img_file, results)
        self.assertIn(nested_img, results)

    def test_find_all_images_nonexistent_path(self):
        """Test find_all_images with nonexistent path"""
        nonexistent = self.test_dir / 'nonexistent'

        results = list(find_all_images(str(nonexistent)))

        self.assertEqual(len(results), 0)

    def test_ibatch_basic(self):
        """Test ibatch basic functionality"""
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        batches = list(ibatch(items, 3))

        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], [1, 2, 3])
        self.assertEqual(batches[1], [4, 5, 6])
        self.assertEqual(batches[2], [7, 8, 9])
        self.assertEqual(batches[3], [10])

    def test_ibatch_exact_multiple(self):
        """Test ibatch with exact multiple batch size"""
        items = [1, 2, 3, 4, 5, 6]
        batches = list(ibatch(items, 3))

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], [1, 2, 3])
        self.assertEqual(batches[1], [4, 5, 6])

    def test_ibatch_empty(self):
        """Test ibatch with empty list"""
        batches = list(ibatch([], 3))

        self.assertEqual(len(batches), 0)

    def test_ibatch_single_item(self):
        """Test ibatch with single item"""
        batches = list(ibatch([42], 5))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0], [42])

    def test_color_formatter(self):
        """Test ColorFormatter functionality"""
        import logging

        formatter = ColorFormatter()

        # Test colorization - check that color codes are present
        debug_result = formatter.colorize('DEBUG', 'test message')
        info_result = formatter.colorize('INFO', 'test message')
        warning_result = formatter.colorize('WARNING', 'test message')
        error_result = formatter.colorize('ERROR', 'test message')
        critical_result = formatter.colorize('CRITICAL', 'test message')

        # Check that ANSI color codes are present
        self.assertIn('\x1b[', debug_result)
        self.assertIn('\x1b[', info_result)
        self.assertIn('\x1b[', warning_result)
        self.assertIn('\x1b[', error_result)
        self.assertIn('\x1b[', critical_result)

        # Test format method
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py', lineno=1, msg='test message', args=(), exc_info=None
        )

        formatted = formatter.format(record)
        self.assertIn('test message', formatted)

    def test_get_logger_tty(self):
        """Test get_logger in TTY environment"""
        with patch('sys.stderr.isatty', return_value=True):
            logger = get_logger('test_tty')

            self.assertEqual(logger.name, 'test_tty')
            self.assertEqual(logger.level, 20)  # INFO level
            self.assertEqual(len(logger.handlers), 1)

    def test_get_logger_non_tty(self):
        """Test get_logger in non-TTY environment"""
        with patch('sys.stderr.isatty', return_value=False), patch('pathlib.Path.mkdir'):
            logger = get_logger('test_non_tty')

            self.assertEqual(logger.name, 'test_non_tty')
            self.assertEqual(logger.level, 20)  # INFO level
            self.assertEqual(len(logger.handlers), 1)

    def test_get_logger_custom_level(self):
        """Test get_logger with custom log level"""
        with patch('sys.stderr.isatty', return_value=True):
            logger = get_logger('test_custom', level=30)  # WARNING level

            self.assertEqual(logger.level, 30)

    def test_print_warn(self):
        """Test print_warn function"""
        with patch('builtins.print') as mock_print:
            print_warn('test warning')
            mock_print.assert_called_once()
            args, kwargs = mock_print.call_args
            self.assertIn('test warning', str(args[0]))
            self.assertEqual(kwargs.get('file'), sys.stderr)

    def test_print_err(self):
        """Test print_err function"""
        with patch('builtins.print') as mock_print:
            print_err('test error')
            mock_print.assert_called_once()
            args, kwargs = mock_print.call_args
            self.assertIn('test error', str(args[0]))
            self.assertEqual(kwargs.get('file'), sys.stderr)

    @patch('platform.system')
    @patch('subprocess.run')
    def test_open_images_windows(self, mock_run, mock_system):
        """Test open_images on Windows"""
        mock_system.return_value = 'Windows'

        paths = ['image1.jpg', 'image2.png']
        open_images(paths)

        mock_run.assert_called_once_with(['explorer', *paths])

    @patch('platform.system')
    @patch('subprocess.run')
    def test_open_images_darwin(self, mock_run, mock_system):
        """Test open_images on macOS"""
        mock_system.return_value = 'Darwin'

        paths = ['image1.jpg', 'image2.png']
        open_images(paths)

        mock_run.assert_called_once_with(['open', *paths])

    @patch('platform.system')
    @patch('subprocess.run')
    def test_open_images_linux(self, mock_run, mock_system):
        """Test open_images on Linux"""
        mock_system.return_value = 'Linux'

        paths = ['image1.jpg', 'image2.png']
        open_images(paths)

        mock_run.assert_called_once_with(['xdg-open', *paths])

    @patch('platform.system')
    @patch('subprocess.run')
    def test_open_images_exception(self, mock_run, mock_system):
        """Test open_images with exception handling"""
        mock_system.return_value = 'Linux'
        mock_run.side_effect = Exception('Command not found')

        with patch('imgsearch.utils.print_err') as mock_print_err:
            open_images(['test.jpg'])
            mock_print_err.assert_called_once()

    def test_path_types(self):
        """Test functions work with both string and Path objects"""
        # Create test image
        img_path = self.test_dir / 'test.jpg'
        Image.new('RGB', (10, 10), color='red').save(img_path)

        # Test with string path
        str_results = list(find_all_images(str(img_path)))
        self.assertEqual(len(str_results), 1)

        # Test with Path object
        path_results = list(find_all_images(img_path))
        self.assertEqual(len(path_results), 1)

        # Results should be identical
        self.assertEqual(str_results[0], path_results[0])
