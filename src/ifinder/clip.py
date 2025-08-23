import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL = 'wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M'


class Clip:
    """CLIP model wrapper"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.device = self.get_device(device)
        self.model, self.processor = self.load_model(model_name)

    @staticmethod
    def get_device(name: str | None = None):
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
        processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        return model, processor

    def embed_images(self, images: list[Image.Image]):
        """Embed images to feature vectors"""
