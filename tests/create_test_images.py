#!/usr/bin/env python3
"""
创建测试图片
"""

from PIL import Image
from pathlib import Path

def create_test_images():
    """创建一些测试图片"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 创建不同颜色的图片
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "orange": (255, 165, 0),
    }
    
    for name, color in colors.items():
        img = Image.new('RGB', (200, 200), color)
        img.save(test_dir / f"{name}.jpg")
        print(f"创建了 {name}.jpg")

if __name__ == "__main__":
    create_test_images()
