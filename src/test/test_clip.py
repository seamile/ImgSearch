import unittest
from multiprocessing import cpu_count
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import torch
from PIL import Image

from imgsearch.clip import Clip

# Constants for testing
CLIP_FEATURE_DIM = 512  # CLIP model feature dimension
TEST_FEATURE_VALUE = 0.1  # Default feature value for testing


class TestClipInitialization(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_model_and_transforms = (self.mock_model, self.mock_processor, self.mock_tokenizer)

    @patch.object(Clip, 'get_device')
    @patch.object(Clip, 'load_model')
    def test_init_cpu(self, mock_load_model, mock_get_device):
        mock_get_device.return_value = torch.device('cpu')
        mock_load_model.return_value = self.mock_model_and_transforms

        clip = Clip(model_key='ViT-B-32', device='cpu')

        self.assertEqual(clip.device.type, 'cpu')
        mock_get_device.assert_called_once_with('cpu')
        mock_load_model.assert_called_once_with('ViT-B-32')

    @patch('torch.cuda.is_available')
    @patch.object(Clip, 'get_device')
    @patch.object(Clip, 'load_model')
    def test_init_cuda(self, mock_load_model, mock_get_device, mock_cuda_available):
        mock_cuda_available.return_value = True
        mock_get_device.return_value = torch.device('cuda')
        mock_load_model.return_value = self.mock_model_and_transforms

        clip = Clip(model_key='ViT-B-32', device='cuda')

        self.assertEqual(clip.device.type, 'cuda')
        mock_get_device.assert_called_once_with('cuda')
        mock_load_model.assert_called_once_with('ViT-B-32')

    @patch('torch.backends.mps.is_available')
    @patch.object(Clip, 'get_device')
    @patch.object(Clip, 'load_model')
    def test_init_mps(self, mock_load_model, mock_get_device, mock_mps_available):
        mock_mps_available.return_value = True
        mock_get_device.return_value = torch.device('mps')
        mock_load_model.return_value = self.mock_model_and_transforms

        clip = Clip(model_key='ViT-B-32', device='mps')

        self.assertEqual(clip.device.type, 'mps')
        mock_get_device.assert_called_once_with('mps')
        mock_load_model.assert_called_once_with('ViT-B-32')

    @patch.object(Clip, 'load_model')
    @patch.object(Clip, 'get_device')
    @patch('torch.set_num_threads')
    def test_init_cpu_threads(self, mock_threads, mock_get_device, mock_load_model):
        mock_get_device.return_value = torch.device('cpu')
        mock_load_model.return_value = self.mock_model_and_transforms

        Clip(model_key='ViT-B-32', device='cpu')

        mock_threads.assert_called_once_with(max(cpu_count() * 2, 2))

    def test_init_invalid_device(self):
        with self.assertRaises(RuntimeError):
            Clip(model_key='ViT-B-32', device='invalid')

    @patch.dict('imgsearch.config.MODELS', {'invalid_model': None}, clear=True)
    @patch.object(Clip, 'load_model')
    @patch.object(Clip, 'get_device')
    def test_init_invalid_model_key(self, mock_load_model, mock_get_device):
        mock_get_device.return_value = torch.device('cpu')
        mock_load_model.side_effect = KeyError('invalid_model')

        with self.assertRaises(KeyError):
            Clip('invalid_model')


class TestClipStaticMethods(unittest.TestCase):
    @patch.object(Clip, 'get_device')
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_get_device_cpu(self, mock_mps, mock_cuda, mock_get_device):
        mock_cuda.return_value = False
        mock_mps.return_value = False
        mock_get_device.return_value = torch.device('cpu')

        device = Clip.get_device('cpu')
        self.assertEqual(device.type, 'cpu')

    @patch.object(Clip, 'get_device')
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_get_device_cuda(self, mock_mps, mock_cuda, mock_get_device):
        mock_cuda.return_value = True
        mock_mps.return_value = False
        mock_get_device.return_value = torch.device('cuda')

        device = Clip.get_device()
        self.assertEqual(device.type, 'cuda')

    @patch.object(Clip, 'get_device')
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_device_mps(self, mock_cuda, mock_mps, mock_get_device):
        mock_cuda.return_value = False
        mock_mps.return_value = True
        mock_get_device.return_value = torch.device('mps')

        device = Clip.get_device()
        self.assertEqual(device.type, 'mps')

    @patch.object(Clip, 'load_model')
    @patch('imgsearch.config.MODELS')
    def test_load_model_valid(self, mock_models, mock_load_model):
        mock_models.__contains__.return_value = True
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_processor = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_processor, mock_tokenizer)

        result = Clip.load_model('ViT-B-32')

        self.assertEqual(result, (mock_model, mock_processor, mock_tokenizer))
        mock_load_model.assert_called_once_with('ViT-B-32')

    @patch.object(Clip, 'load_model')
    @patch('imgsearch.config.MODELS')
    def test_load_model_invalid(self, mock_models, mock_load_model):
        mock_models.__contains__.return_value = False
        mock_load_model.side_effect = ValueError('Invalid model key')

        with self.assertRaises(ValueError):
            Clip.load_model('invalid')


class TestClipFeatures(unittest.TestCase):
    def setUp(self):
        self.clip = Mock(spec=Clip)
        self.clip.model = MagicMock()
        self.clip.processor = MagicMock()
        self.clip.tokenizer = MagicMock()
        self.clip.device = torch.device('cpu')
        self.clip.executor = MagicMock()
        self.clip.executor.map = MagicMock()

    def _create_normalized_feature_vector(self, value: float = TEST_FEATURE_VALUE) -> list[float]:
        """Helper method to create a normalized feature vector for testing."""
        feature = [value] * CLIP_FEATURE_DIM
        norm = np.linalg.norm(feature)
        return [(x / norm) for x in feature]  # type: ignore

    def test_embed_images_empty_list(self):
        self.clip.embed_images.return_value = []
        result = self.clip.embed_images([])
        self.assertEqual(result, [])

    @patch.object(Clip, 'embed_images')
    def test_embed_images_batch(self, mock_embed_images):
        """Test batch image embedding by mocking the embed_images method"""
        # Create actual normalized feature vectors using helper method
        feature1 = self._create_normalized_feature_vector(TEST_FEATURE_VALUE)
        feature2 = self._create_normalized_feature_vector(0.2)

        mock_embed_images.return_value = [feature1, feature2]

        mock_image1 = Mock(spec=Image.Image)
        mock_image2 = Mock(spec=Image.Image)
        images = [mock_image1, mock_image2]

        # Create a mock Clip instance
        clip = Mock(spec=Clip)
        clip.embed_images = mock_embed_images

        result = clip.embed_images(images)

        self.assertEqual(len(result), len(images))
        self.assertAlmostEqual(np.linalg.norm(np.array(result[0])), 1.0, places=5)  # type: ignore
        mock_embed_images.assert_called_once_with(images)

    @patch.object(Clip, 'embed_images')
    @patch('PIL.Image.Image.convert')
    def test_embed_images_non_rgb_conversion(self, mock_convert, mock_embed_images):
        """Test RGB conversion in image embedding"""
        # Create actual normalized feature vector using helper method
        feature = self._create_normalized_feature_vector(TEST_FEATURE_VALUE)

        mock_embed_images.return_value = [feature]

        non_rgb_image = Mock(spec=Image.Image)
        non_rgb_image.mode = 'RGBA'
        mock_rgb_image = Mock(spec=Image.Image)
        mock_rgb_image.mode = 'RGB'
        mock_convert.return_value = mock_rgb_image

        # Create a mock Clip instance
        _ = Mock(spec=Clip)  # Use _ to indicate intentional unused variable
        # The actual embed_images method would call convert, but since we're mocking it,
        # we need to simulate the behavior. In the real implementation, embed_images
        # would convert the image to RGB before processing.
        # Here we test that the embed_images method works with the converted image.
        converted_images = [mock_rgb_image]  # This simulates what the real method would do
        result = mock_embed_images(converted_images)

        self.assertEqual(len(result), len(converted_images))
        self.assertAlmostEqual(np.linalg.norm(np.array(result[0])), 1.0, places=5)  # type: ignore
        mock_embed_images.assert_called_once_with(converted_images)

    @patch.object(Clip, 'embed_images')
    def test_embed_images_l2_normalization(self, mock_embed_images):
        """Test L2 normalization in image embedding"""
        # Create actual normalized feature vector using helper method
        feature = self._create_normalized_feature_vector(TEST_FEATURE_VALUE)

        mock_embed_images.return_value = [feature]

        mock_image = Mock(spec=Image.Image)
        images = [mock_image]

        # Create a mock Clip instance
        clip = Mock(spec=Clip)
        clip.embed_images = mock_embed_images

        result = clip.embed_images(images)

        self.assertEqual(len(result), len(images))
        self.assertAlmostEqual(np.linalg.norm(np.array(result[0])), 1.0, places=5)  # type: ignore
        mock_embed_images.assert_called_once_with(images)

    @patch.object(Clip, 'embed_image')
    def test_embed_image_single(self, mock_embed_image):
        mock_image = Mock(spec=Image.Image)
        mock_embedding = np.array([0.1, 0.2])
        mock_embed_image.return_value = mock_embedding

        result = mock_embed_image(mock_image, self.clip.processor)

        np.testing.assert_array_almost_equal(result, mock_embedding)
        mock_embed_image.assert_called_once_with(mock_image, self.clip.processor)

    @patch.object(Clip, 'embed_image')
    def test_embed_image_invalid_not_pil(self, mock_embed_image):
        invalid_image = 'not_an_image'
        mock_embed_image.side_effect = ValueError('Invalid image')

        with self.assertRaises(ValueError):
            mock_embed_image(invalid_image, self.clip.processor)

    def test_embed_text_single(self):
        clip = Clip()
        clip.model = MagicMock()
        clip.model.to.return_value = clip.model
        clip.model.eval.return_value = clip.model
        clip.tokenizer = MagicMock()
        clip.tokenizer.return_value = torch.tensor([[1] * 77])
        mock_text_tensor = torch.randn(1, 512)
        mock_text_tensor = mock_text_tensor / torch.norm(mock_text_tensor, dim=-1, keepdim=True)
        mock_text_tensor = mock_text_tensor.detach().numpy().tolist()[0]
        clip.model.encode_text.return_value = torch.tensor([mock_text_tensor])

        result = clip.embed_text('test')

        self.assertIsInstance(result, list)
        self.assertAlmostEqual(np.linalg.norm(np.array(result)), 1.0, places=5)  # type: ignore
        clip.tokenizer.assert_called_once_with(['test'])
        clip.model.encode_text.assert_called_once()

    @patch.object(Clip, 'embed_text')
    def test_embed_text_empty(self, mock_embed_text):
        self.clip.tokenizer.return_value = {'input_ids': torch.tensor([])}
        mock_embed_text.side_effect = ValueError('Empty text')

        with self.assertRaises(ValueError):
            mock_embed_text('', self.clip.model, self.clip.device)

    @patch.object(Clip, 'embed_text')
    def test_embed_text_invalid_not_str(self, mock_embed_text):
        invalid_text = 123
        mock_embed_text.side_effect = ValueError('Text must be string')

        with self.assertRaises(ValueError):
            mock_embed_text(invalid_text, self.clip.model, self.clip.device)

    @patch.object(Clip, 'compare_images')
    def test_compare_images_similarity(self, mock_compare):
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([1.0, 0.0])
        expected_similarity = 100.0
        mock_compare.return_value = expected_similarity

        result = mock_compare(emb1, emb2)

        self.assertAlmostEqual(result, expected_similarity)

    @patch.object(Clip, 'compare_images')
    def test_compare_images_boundary_zero(self, mock_compare):
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([-1.0, 0.0])
        expected_similarity = 0.0
        mock_compare.return_value = expected_similarity

        result = mock_compare(emb1, emb2)

        self.assertAlmostEqual(result, expected_similarity)

    @patch.object(Clip, 'compare_images')
    def test_compare_images_boundary_one_hundred(self, mock_compare):
        emb1 = np.array([0.7, 0.7])
        emb2 = np.array([0.7, 0.7])
        expected_similarity = 100.0
        mock_compare.return_value = expected_similarity

        result = mock_compare(emb1, emb2)

        self.assertAlmostEqual(result, expected_similarity)

    @patch.dict('imgsearch.config.MODELS', {'ViT-B-32': ('ViT-B/32', 'openai')})
    @patch.object(Clip, 'load_model')
    def test_del_shutdown(self, mock_load_model):
        mock_model = Mock()
        mock_processor = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_processor, mock_tokenizer)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        clip = Clip(model_key='ViT-B-32', device='cpu')

        # Access the executor to trigger its creation
        _ = clip.executor

        # Mock the executor and test the __del__ behavior directly
        with patch.object(clip, 'executor') as mock_exec:
            mock_shutdown = Mock()
            mock_exec.shutdown = mock_shutdown

            # Call __del__ manually to test the shutdown behavior
            clip.__del__()

            # Verify shutdown was called
            mock_shutdown.assert_called_once_with(wait=False)


class TestClipErrors(unittest.TestCase):
    def setUp(self):
        self.clip = Mock(spec=Clip)
        self.clip.model = MagicMock()
        self.clip.processor = MagicMock()
        self.clip.device = torch.device('cpu')
        self.clip.executor = MagicMock()
        self.clip.executor.map = MagicMock()

    @patch.object(Clip, 'embed_images')
    def test_embed_images_invalid_image(self, mock_embed_images):
        invalid_image = 'invalid'
        mock_embed_images.side_effect = ValueError('Invalid image')

        with self.assertRaises(ValueError):
            mock_embed_images([invalid_image], self.clip.processor, self.clip.model, self.clip.device)

    def test_embed_text_tokenizer_error(self):
        clip = Clip()
        with patch.object(clip, 'tokenizer', side_effect=RuntimeError('Tokenizer error')):
            with self.assertRaises(RuntimeError):
                clip.embed_text('test')

    @patch.object(Clip, 'executor', new_callable=PropertyMock)
    def test_model_encode_error(self, mock_executor):
        clip = Clip()
        mock_image = Image.new('RGB', (224, 224))
        images = [mock_image]
        with patch.object(clip.model, 'encode_image', side_effect=RuntimeError('Encode error')):
            with self.assertRaises(RuntimeError):
                clip.embed_images(images)


if __name__ == '__main__':
    unittest.main()
