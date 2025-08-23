"""
iFinder - Image Search Tool

A CLIP-based image search tool supporting both image-to-image and text-to-image search.
"""

# Export main classes for module usage
from .clip import Clip
from .storage import ImgBase

__all__ = ['Clip', 'ImgBase']
