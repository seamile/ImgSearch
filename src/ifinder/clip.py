import logging

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ifinder.consts import DEFAULT_MODEL
from ifinder.utils import Feature

# Disable transformers warnings
for name, logger in logging.Logger.manager.loggerDict.items():
    if name.startswith('transformers.') and isinstance(logger, logging.Logger):
        logger.setLevel(logging.ERROR)


class Clip:
    """CLIP model wrapper"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.device = self.get_device(device)
        self.model, self.processor = self.load_model(model_name)
        # Move model to device and ensure proper dtype
        self.model = self.model.to(self.device)  # type: ignore

        # For MPS, ensure model is in float32
        if self.device.type == 'mps':
            self.model = self.model.float()

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
    def load_model(model_name: str = DEFAULT_MODEL) -> tuple[CLIPModel, CLIPProcessor]:
        """Load CLIP model and processor"""
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor

    def embed_images(self, images: list[Image.Image]) -> list[Feature]:
        """Embed a list of images to feature vectors"""
        if not images:
            return []

        # Process images
        inputs = self.processor(images=images, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().tolist()

    def embed_image(self, image: Image.Image) -> Feature:
        """Embed a single image to a feature vector"""
        return self.embed_images([image])[0]

    def embed_text(self, text: str) -> Feature:
        """Embed a single text string to a feature vector"""
        # Process text
        inputs = self.processor(text=[text], return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get text features
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().tolist()[0]
