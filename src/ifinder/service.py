from pathlib import Path

from PIL import Image

from .clip import Clip
from .storage import DEFAULT_INDEX, ImgBase
from .utils import find_all_images, is_image, print_err


class IFinderService:
    """iFinder 核心服务类"""

    def __init__(self, db_path: Path = DEFAULT_INDEX, model_name: str | None = None):
        self.db_path = db_path
        self.clip = Clip(model_name) if model_name else Clip()
        self.imgbase: ImgBase | None = None

    def load_database(self) -> bool:
        """加载数据库"""
        try:
            if self.db_path.exists():
                self.imgbase = ImgBase.load_db(self.db_path)
                return True
            else:
                self.imgbase = ImgBase()
                return False
        except Exception as e:
            print_err(f'加载数据库失败: {e}')
            self.imgbase = ImgBase()
            return False

    def save_database(self) -> bool:
        """保存数据库"""
        try:
            if self.imgbase is not None:
                self.imgbase.save(self.db_path)
                return True
            return False
        except Exception as e:
            print_err(f'保存数据库失败: {e}')
            return False

    def add_images_from_paths(self, paths: list[str | Path], show_progress: bool = True) -> int:
        """从路径列表添加图片到索引"""
        if self.imgbase is None:
            self.load_database()

        # 找到所有图片文件
        image_paths = list(find_all_images(paths))
        if not image_paths:
            print_err('未找到任何图片文件')
            return 0

        added_count = 0
        batch_size = 32  # 批量处理图片

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            valid_paths = []

            # 加载批量图片
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    valid_paths.append(img_path)
                except Exception as e:
                    print_err(f'无法加载图片 {img_path}: {e}')
                    continue

            if not batch_images:
                continue

            try:
                # 提取特征
                features = self.clip.embed_images(batch_images)

                # 添加到索引
                self.imgbase.add_images(valid_paths, features)
                added_count += len(valid_paths)

                if show_progress:
                    print(f'已处理 {min(i + batch_size, len(image_paths))}/{len(image_paths)} 张图片')

            except Exception as e:
                print_err(f'处理图片批次失败: {e}')
                continue

        return added_count

    def search_by_image(self, image_path: str | Path, k: int = 10) -> list[tuple[str, float]]:
        """通过图片搜索相似图片"""
        if self.imgbase is None:
            self.load_database()

        try:
            # 加载查询图片
            img = Image.open(image_path).convert('RGB')

            # 提取特征
            feature = self.clip.embed_image(img)

            # 搜索
            return self.imgbase.search(feature, k)

        except Exception as e:
            print_err(f'图片搜索失败: {e}')
            return []

    def search_by_text(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        """通过文本搜索图片"""
        if self.imgbase is None:
            self.load_database()

        try:
            # 提取文本特征
            feature = self.clip.embed_text(text)

            # 搜索
            return self.imgbase.search(feature, k)

        except Exception as e:
            print_err(f'文本搜索失败: {e}')
            return []

    def search(self, query: str | Path, k: int = 10) -> list[tuple[str, float]]:
        """智能搜索：自动判断是图片路径还是文本"""
        # 如果是路径且是图片文件，则进行图片搜索
        if isinstance(query, (str, Path)):
            query_path = Path(query) if isinstance(query, str) else query
            if query_path.exists() and is_image(query_path):
                return self.search_by_image(query_path, k)

        # 否则进行文本搜索
        query_text = str(query)
        return self.search_by_text(query_text, k)
