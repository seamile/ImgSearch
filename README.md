# iFinder

iFinder 是一款图片查找工具，可以用来以图搜图，也可以通过文字描述搜索图片。

您可以直接在命令行中使用，也可以把它融入到其他系统中作为一个图片搜索模块。

本项目使用 TinyCLIP 模型提取图片特征，数据存储和搜索使用 Hnswlib。

## 使用方法

### 1. 构建索引

初次使用时 iFinder 时需先构建图片索引，将您系统中的图片导入到索引中。

```shell
ifinder -a IMG_FILES_or_DIRS ...
```

### 2. 搜索图片

- 以图搜图:

    ```shell
    ifinder IMAGE_PATH
    ```

- 以关键词搜图:

    ```shell
    ifinder KEYWORDS
    ```

- 其他可选项：

    - `-d DB_PATH` 指定数据库文件的路径
    - `-n NUM` 指定搜索结果数量, 默认为 10
    - `-m MODEL` CLIP 模型名称

### 3. 作为模块导入

```python
from ifinder import Clip, ImgBase
from pathlib import Path
from PIL import Image

# 创建实例
clip = Clip()
imgbase = ImgBase(db_path=Path("my_images.db"))

# 添加图片到索引
image_paths = [Path("/path/to/image1.jpg"), Path("/path/to/image2.jpg")]
images = [Image.open(path).convert('RGB') for path in image_paths]
features = clip.embed_images(images)
imgbase.add_images(image_paths, features)
imgbase.save()

# 以图搜图
query_img = Image.open("/path/to/query.jpg").convert('RGB')
query_feature = clip.embed_image(query_img)
results = imgbase.search(query_feature, k=10)
for path, similarity in results:
    print(f"{path} (相似度: {similarity:.3f})")

# 文本搜索
text_feature = clip.embed_text("red flower")
text_results = imgbase.search(text_feature, k=10)
for path, similarity in text_results:
    print(f"{path} (相似度: {similarity:.3f})")
```
