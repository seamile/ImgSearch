#!/usr/bin/env python3
"""
iFinder 基本功能测试
"""

import tempfile
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ifinder import Clip, ImgBase


def create_test_image(color: tuple[int, int, int], size: tuple[int, int] = (100, 100)) -> Image.Image:
    """创建测试图片"""
    return Image.new('RGB', size, color)


def test_clip_functionality():
    """测试 CLIP 功能"""
    print("测试 CLIP 功能...")
    
    clip = Clip()
    
    # 创建测试图片
    red_img = create_test_image((255, 0, 0))
    blue_img = create_test_image((0, 0, 255))
    
    # 测试批量图片嵌入
    features = clip.embed_images([red_img, blue_img])
    assert len(features) == 2
    assert len(features[0]) == 512  # 特征维度
    print("✅ 批量图片嵌入测试通过")
    
    # 测试单张图片嵌入
    single_feature = clip.embed_image(red_img)
    assert len(single_feature) == 512
    print("✅ 单张图片嵌入测试通过")
    
    # 测试文本嵌入
    text_feature = clip.embed_text("red color")
    assert len(text_feature) == 512
    print("✅ 文本嵌入测试通过")


def test_imgbase_functionality():
    """测试 ImgBase 功能"""
    print("测试 ImgBase 功能...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "test.db"
        
        # 创建 ImgBase 实例
        imgbase = ImgBase(db_path=db_path)
        
        # 测试添加图片
        test_paths = [temp_path / "red.jpg", temp_path / "blue.jpg"]
        test_features = [[0.1] * 512, [0.2] * 512]
        
        imgbase.add_images(test_paths, test_features)
        assert imgbase.current_id == 2
        print("✅ 添加图片测试通过")
        
        # 测试搜索
        results = imgbase.search([0.1] * 512, k=2)
        assert len(results) == 2
        print("✅ 搜索功能测试通过")
        
        # 测试保存和加载
        imgbase.save()
        assert db_path.exists()
        print("✅ 保存功能测试通过")
        
        # 测试加载
        new_imgbase = ImgBase(db_path=db_path)
        assert new_imgbase.current_id == 2
        assert len(new_imgbase.idx_mapping) == 2
        print("✅ 加载功能测试通过")


def test_integration():
    """集成测试"""
    print("集成测试...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "test.db"
        
        # 创建实例
        clip = Clip()
        imgbase = ImgBase(db_path=db_path)
        
        # 创建测试图片
        red_img = create_test_image((255, 0, 0))
        blue_img = create_test_image((0, 0, 255))
        
        red_path = temp_path / "red.jpg"
        blue_path = temp_path / "blue.jpg"
        
        red_img.save(red_path)
        blue_img.save(blue_path)
        
        # 提取特征并添加到索引
        features = clip.embed_images([red_img, blue_img])
        imgbase.add_images([red_path, blue_path], features)
        
        # 测试图片搜索
        query_feature = clip.embed_image(red_img)
        results = imgbase.search(query_feature, k=2)
        
        assert len(results) == 2
        # 第一个结果应该是红色图片本身，相似度最高
        assert results[0][1] > results[1][1]
        print("✅ 集成测试通过")


if __name__ == "__main__":
    try:
        test_clip_functionality()
        test_imgbase_functionality()
        test_integration()
        print("\n🎉 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
