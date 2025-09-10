import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

from imgsearch.consts import DEFAULT_MODEL, PRETRAINED
from imgsearch.utils import Feature, cpu_count

# Disable transformers warnings
for name, logger in logging.Logger.manager.loggerDict.items():
    if name.startswith('transformers.') and isinstance(logger, logging.Logger):
        logger.setLevel(logging.ERROR)


class Clip:
    """CLIP model wrapper"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.device = self.get_device(device)
        self.model, self.processor, self.tokenizer = self.load_model(model_name)
        # Move model to device and ensure proper dtype
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()

        if self.device.type == 'cpu':
            # Set number of threads to avoid slowdown
            torch.set_num_threads(max(cpu_count(), 2))

        # For MPS, ensure model is in float32
        if self.device.type == 'mps':
            self.model = self.model.float()

        # Thread pool for concurrent preprocessing
        self._executor: ThreadPoolExecutor | None = None

    def __del__(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max(cpu_count(), 2))
        return self._executor

    @staticmethod
    def get_device(name: str | None = None) -> torch.device:
        """Get device type"""
        if name is not None:
            return torch.device(name)
        elif torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    @staticmethod
    def load_model(model_name: str = DEFAULT_MODEL, pretrained: str = PRETRAINED):
        """Load CLIP model and processor"""
        model, _, processor = create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = get_tokenizer(model_name)
        return model, processor, tokenizer  # type: ignore

    def embed_images(self, images: list[Image.Image]) -> list[Feature]:
        """Embed a list of images to feature vectors"""
        if not images:
            return []

        # make sure all images are RGB mode
        for i, img in enumerate(images):
            if img.mode != 'RGB':
                images[i] = img.convert('RGB')

        # Concurrent preprocessing for I/O bound operations
        if len(images) > 1:
            img_tensors = list(self.executor.map(self.processor, images))  # type: ignore
        else:
            img_tensors = [self.processor(images[0])]  # type: ignore

        batch_tensor = torch.stack(img_tensors)  # type: ignore

        # Get image features
        device_type, non_blocking = ('cuda', True) if self.device.type == 'cuda' else ('cpu', False)
        with torch.no_grad(), torch.autocast(device_type=device_type):
            img_features = self.model.encode_image(batch_tensor.to(self.device, non_blocking=non_blocking))  # type: ignore
            # Normalize features
            img_features /= img_features.norm(dim=-1, keepdim=True)

        return [f.cpu().numpy().tolist() for f in img_features]

    def embed_image(self, image: Image.Image) -> Feature:
        """Embed a single image to a feature vector"""
        return self.embed_images([image])[0]

    def embed_text(self, *text: str) -> Feature:
        """Embed a single text string to a feature vector"""
        if not text:
            return []

        # Process text
        text_tensor = self.tokenizer(list(text))

        # Get text features
        device_type, non_blocking = ('cuda', True) if self.device.type == 'cuda' else ('cpu', False)
        with torch.no_grad(), torch.autocast(device_type=device_type):
            text_tensor = text_tensor.to(self.device, non_blocking=non_blocking)
            text_features = self.model.encode_text(text_tensor)  # type: ignore
            # Normalize features
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().tolist()[0]

    def compare_images(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compare similarity between two images"""
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')

        feature1, feature2 = self.embed_images([img1, img2])
        similarity = np.dot(feature1, feature2)
        return round(float(similarity) * 100, 2)
