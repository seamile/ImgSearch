"""CLIP Model Wrapper Module

This module provides a high-level wrapper around OpenCLIP models for multimodal
(image/text) embedding. Supports multiple model variants, automatic device
selection (CUDA/MPS/CPU), batch processing, and thread-safe concurrent
preprocessing. Features are normalized to unit length for cosine similarity.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

import numpy as np
import torch
from PIL import Image

from imgsearch import config as cfg
from imgsearch.utils import Feature, cpu_count
from tinyclip import create_model_and_transforms, get_tokenizer


class Clip:
    """Wrapper for OpenCLIP models supporting image/text embedding and similarity.

    Handles model loading, device placement (auto-detect CUDA/MPS/CPU), tensor
    optimization, and batch processing. Features are L2-normalized for cosine
    similarity computation.

    Thread-safe: uses ThreadPoolExecutor for concurrent image preprocessing.
    Destructor shuts down executor to prevent resource leaks.

    Example:
        clip = Clip('ViT-45LY')  # Load default model (ViT-45LY)
        features = clip.embed_images([img1, img2])  # Batch embed
        sim = clip.compare_images(img1, img2)  # 0-100% similarity
    """

    def __init__(self, model_key: str = cfg.DEFAULT_MODEL_KEY, device: str | None = None) -> None:
        """Initialize CLIP wrapper with model loading and device setup.

        Loads model/transforms/tokenizer from OpenCLIP, moves to optimal device,
        sets evaluation mode, and configures threading for CPU/MPS.

        Args:
            model_key (str): Model variant key from cfg.MODELS.
                Defaults to cfg.DEFAULT_MODEL_KEY ('ViT-45LY').
            device (str | None): Override device ('cuda', 'mps', 'cpu').
                Defaults to auto-detection.
        """
        self.device = self.get_device(device)
        self.model, self.processor, self.tokenizer = self.load_model(model_key)

        # Move model to device; ensure float32 for MPS compatibility
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()  # Disable training-specific layers

        if self.device.type == 'cpu':
            # Optimize CPU threading to prevent slowdowns
            torch.set_num_threads(max(cpu_count() * 2, 2))

        # MPS requires explicit float32 (avoids precision issues)
        if self.device.type == 'mps':
            self.model = self.model.float()

    def __del__(self):
        if 'executor' in self.__dict__:
            self.executor.shutdown(wait=False)
            del self.executor

    @cached_property
    def executor(self) -> ThreadPoolExecutor:
        """Get thread pool executor for concurrent preprocessing (lazy init).

        Uses CPU count for workers to parallelize I/O-heavy image transforms.
        Ensures thread-safety for multi-threaded embedding calls.

        Returns:
            ThreadPoolExecutor: Configured executor instance.
        """
        return ThreadPoolExecutor(max_workers=max(cpu_count(), 2))

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
    def load_model(model_key: str):
        """Load CLIP model and processor"""
        model_name, pretrained, _ = cfg.MODELS[model_key]
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

        img_features = img_features.cpu().float()
        return [f.numpy().tolist() for f in img_features]

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

        text_features = text_features.cpu().float()
        return text_features.numpy().tolist()[0]

    def compare_images(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compare similarity between two images"""
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')

        feature1, feature2 = self.embed_images([img1, img2])
        similarity = np.dot(feature1, feature2)
        return round(float(similarity) * 100, 2)
