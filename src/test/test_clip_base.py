"""
Base unit tests for clip.py module - focusing on core functionality with mocks
"""

import unittest
from unittest.mock import Mock, patch

from imgsearch.clip import Clip


@patch('transformers.CLIPProcessor.from_pretrained', return_value=Mock(name='mock_processor'))
@patch('transformers.CLIPModel.from_pretrained', return_value=Mock(name='mock_model'))
class TestClipBase(unittest.TestCase):
    """Base test class for Clip model wrapper - using mocked dependencies"""

    def test_get_device_explicit(self, mock_get_model, mock_get_processor):
        """Test get_device with explicit device name"""
        device = Clip.get_device('cpu')
        self.assertEqual(device.type, 'cpu')

    def test_get_device_cpu_fallback(self, mock_get_model, mock_get_processor):
        """Test get_device fallback to CPU"""
        with (
            patch('torch.cuda.is_available', return_value=False),
            patch('torch.backends.mps.is_available', return_value=False),
        ):
            device = Clip.get_device()
            self.assertEqual(device.type, 'cpu')

    def test_init_basic(self, mock_get_model, mock_get_processor):
        """Test basic initialization"""
        clip = Clip(device='cpu')
        self.assertEqual(clip.device.type, 'cpu')

    def test_init_default_device(self, mock_get_model, mock_get_processor):
        """Test initialization with default device selection"""
        with (
            patch('torch.cuda.is_available', return_value=False),
            patch('torch.backends.mps.is_available', return_value=False),
        ):
            clip = Clip()
            self.assertEqual(clip.device.type, 'cpu')

    def test_init_cuda_device(self, mock_get_model, mock_get_processor):
        """Test initialization with CUDA device - skip if not available"""
        with (
            patch('torch.cuda.is_available', return_value=True),
            patch('torch.backends.mps.is_available', return_value=False),
        ):
            clip_1 = Clip()
            self.assertEqual(clip_1.device.type, 'cuda')

            clip_2 = Clip(device='cpu')  # Force CPU to avoid CUDA issues
            self.assertEqual(clip_2.device.type, 'cpu')

    def test_init_mps_device(self, mock_get_model, mock_get_processor):
        """Test initialization with MPS device"""
        with (
            patch('torch.cuda.is_available', return_value=False),
            patch('torch.backends.mps.is_available', return_value=True),
        ):
            clip = Clip()
            self.assertEqual(clip.device.type, 'mps')

    def test_load_model_mock(self, mock_get_model, mock_get_processor):
        """Test model loading with mocks"""
        model, processor = Clip.load_model('test-model')
        mock_get_model.assert_called_once_with('test-model')
        mock_get_processor.assert_called_once_with('test-model')

        self.assertEqual(model, mock_get_model())
        self.assertEqual(processor, mock_get_processor())

    def test_embed_images_empty_list(self, mock_get_model, mock_get_processor):
        """Test embed_images with empty list"""
        clip = Clip(device='cpu')
        result = clip.embed_images([])
        self.assertEqual(result, [])

    def test_torch_set_num_threads(self, mock_get_model, mock_get_processor):
        """Test that torch.set_num_threads is called during initialization"""
        with patch('torch.set_num_threads') as mock_set_num_threads:
            Clip(device='cpu')
            mock_set_num_threads.assert_called_once()

    @patch('torch.set_num_threads')
    def test_clip_initialization_flow(self, mock_set_threads, mock_get_model, mock_get_processor):
        """Test the complete initialization flow"""
        clip = Clip(device='cpu')
        mock_set_threads.assert_called_once()

        self.assertEqual(clip.device.type, 'cpu')
        self.assertEqual(clip.processor, mock_get_processor())
