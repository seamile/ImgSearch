"""
CLIP feature extraction tests for clip.py module - testing CLIP model functionality
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from pixa.clip import Clip


@patch('transformers.CLIPModel.from_pretrained', return_value=Mock(name='mock_model'))
@patch('transformers.CLIPProcessor.from_pretrained', return_value=Mock(name='mock_processor'))
class TestClipFeature(unittest.TestCase):
    """CLIP feature extraction test class for Clip model wrapper"""

    def test_embed_images_with_real_tensor_operations(self, mock_get_processor, mock_get_model):
        """Test embed_images with real tensor operations"""
        # Mock model and processor
        mock_model = mock_get_model.return_value
        mock_model.to.return_value = mock_model
        mock_processor = mock_get_processor.return_value

        # Mock processor return
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}

        # Create a real tensor for testing normalization
        features = torch.randn(1, 512)
        mock_model.get_image_features.return_value = features

        clip = Clip(device='cpu')

        # Create test image
        image = Image.new('RGB', (100, 100), color='red')
        result = clip.embed_images([image])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 512)
        self.assertIsInstance(result[0], list)
        # Check that the result is normalized (approximately unit length)
        result_array = np.array(result[0])
        norm = np.linalg.norm(result_array)
        self.assertLess(abs(norm - 1.0), 0.01)  # Should be close to 1.0

    def test_embed_image_single(self, mock_get_processor, mock_get_model):
        """Test embed_image with single image"""
        # Get the actual mock instances created by the decorators
        mock_model = mock_get_model.return_value
        mock_model.to.return_value = mock_model
        mock_processor = mock_get_processor.return_value

        # Mock processor return
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}

        # Create a proper tensor that supports the operations
        features = torch.randn(1, 512)
        mock_model.get_image_features.return_value = features

        clip = Clip(device='cpu')

        image = Image.new('RGB', (100, 100), color='blue')
        result = clip.embed_image(image)

        self.assertEqual(len(result), 512)
        self.assertIsInstance(result, list)

    def test_embed_text(self, mock_get_processor, mock_get_model):
        """Test embed_text with text input"""
        # Get the actual mock instances created by the decorators
        mock_model = mock_get_model.return_value
        mock_model.to.return_value = mock_model
        mock_processor = mock_get_processor.return_value

        # Mock processor return
        mock_processor.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10),
        }

        # Create a proper tensor that supports the operations
        features = torch.randn(1, 512)
        mock_model.get_text_features.return_value = features

        clip = Clip(device='cpu')

        result = clip.embed_text('test text')

        self.assertEqual(len(result), 512)
        self.assertIsInstance(result, list)
        # Check that the result is normalized
        result_array = np.array(result)
        norm = np.linalg.norm(result_array)
        self.assertLess(abs(norm - 1.0), 0.01)

    def test_compare_images(self, mock_get_processor, mock_get_model):
        """Test compare_images functionality"""
        # Get the actual mock instances created by the decorators
        mock_model = mock_get_model.return_value
        mock_model.to.return_value = mock_model
        mock_processor = mock_get_processor.return_value

        # Mock processor return
        mock_processor.return_value = {'pixel_values': torch.randn(2, 3, 224, 224)}

        # Create a proper tensor that supports the operations
        features = torch.tensor([[1.0, 0.0] * 256, [0.8, 0.6] * 256], dtype=torch.float32)
        mock_model.get_image_features.return_value = features

        clip = Clip(device='cpu')

        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        similarity = clip.compare_images(img1, img2)

        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 100.0)

    def test_similarity_calculation_logic(self, mock_get_processor, mock_get_model):
        """Test the similarity calculation logic with known values"""
        # Mock model and processor
        mock_model = mock_get_model.return_value
        mock_model.to.return_value = mock_model
        mock_processor = mock_get_processor.return_value

        # Mock processor return
        mock_processor.return_value = {'pixel_values': torch.randn(2, 3, 224, 224)}

        # Create identical normalized features (should give 100% similarity)
        identical_features = torch.tensor([[1.0, 0.0] * 256, [1.0, 0.0] * 256], dtype=torch.float32)
        identical_features = identical_features / identical_features.norm(dim=-1, keepdim=True)
        mock_model.get_image_features.return_value = identical_features

        clip = Clip(device='cpu')

        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='red')

        similarity = clip.compare_images(img1, img2)

        # Should be close to 100% for identical features
        self.assertGreater(similarity, 99.0)
        self.assertLessEqual(similarity, 100.0)

    def test_compare_images_rgba_conversion(self, mock_get_processor, mock_get_model):
        """Test compare_images with RGBA image conversion"""
        # Mock model and processor
        mock_model = mock_get_model.return_value
        mock_model.to.return_value = mock_model
        mock_processor = mock_get_processor.return_value

        # Mock processor return
        mock_processor.return_value = {'pixel_values': torch.randn(2, 3, 224, 224)}

        # Create a proper tensor that supports the operations
        features = torch.tensor([[0.9, 0.1] * 256, [0.8, 0.6] * 256], dtype=torch.float32)
        mock_model.get_image_features.return_value = features

        clip = Clip(device='cpu')

        # Create RGBA images
        img1 = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img2 = Image.new('RGBA', (100, 100), color=(0, 255, 0, 128))

        similarity = clip.compare_images(img1, img2)

        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 100.0)

    def test_mps_float_conversion(self, mock_get_processor, mock_get_model):
        """Test MPS float conversion for MPS device"""
        mock_model = mock_get_model.return_value
        mock_model.to.return_value = mock_model
        mock_model.float = Mock(return_value=mock_model)

        with (
            patch('torch.cuda.is_available', return_value=False),
            patch('torch.backends.mps.is_available', return_value=True),
            patch.object(mock_model, 'to') as mock_to,
        ):
            mock_to.return_value = mock_model
            clip = Clip()
            self.assertEqual(clip.device.type, 'mps')


if __name__ == '__main__':
    unittest.main()
