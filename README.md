# iFinder

iFinder 是一款图片搜索引擎，可以用来以图搜图，也可以通过文字描述搜索图片。您可以直接在命令行中使用，也可以把它融入到其他系统中作为一个图片搜索模块。

本项目使用 TinyCLIP 模型提取图片特征，数据存储和搜索使用 HNSWlib。程序整体采用 C-S 架构，使用前需要先启动服务，然后通过客户端进行操作。

## 使用方法

### 1. 服务管理

```shell
# 前台启动（用于调试）
ifinder -s start

# 后台启动（守护进程模式）
ifinder -s start -D

# 停止服务
ifinder -s stop

# 重启服务
ifinder -s restart

# 查看服务状态
ifinder -s status
```

### 2. 添加图片到图像索引

```shell
ifinder -a /path/to/images
```

### 3. 搜索图片

- 以图搜图:

    ```shell
    ifinder /path/to/query_image.jpg
    ```

- 以关键词搜图:

    ```shell
    ifinder "red flower"
    ```

### 4. 其他功能

- 查看数据库信息:

    ```shell
    ifinder -i
    ```

- 比较两张图片的相似度:

    ```shell
    ifinder -c image1.jpg image2.jpg
    ```

- 清空数据库:

    ```shell
    ifinder -C
    ```

### 5. 其他可选参数

- `-d DB_DIR`   指定数据库目录路径（默认: ~/.ifinder）
- `-D`          后台运行服务（配合 -s start/restart 使用）
- `-l {path,name}` 标签命名方式：path=绝对路径，name=文件名
- `-m MODEL`    CLIP 模型名称
- `-n NUM`      搜索结果数量（默认: 10）

### 6. 作为模块导入

```python
from ifinder.clip import Clip
from ifinder.storage import VectorDB
from pathlib import Path
from PIL import Image

# 创建实例
clip = Clip()
db = VectorDB(db_dir=Path("~/.ifinder").expanduser())

# 添加图片到索引
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
images = [Image.open(path).convert('RGB') for path in image_paths]
features = clip.embed_images(images)
labels = [str(Path(path).resolve()) for path in image_paths]  # 使用绝对路径作为标签
db.add_items(labels, features)
db.save()

# 以图搜图
query_img = Image.open("/path/to/query.jpg").convert('RGB')
query_feature = clip.embed_image(query_img)
results = db.search(query_feature, k=10)
for path, similarity in results:
    print(f"{path} (相似度: {similarity}%)")

# 文本搜索
text_feature = clip.embed_text("red flower")
text_results = db.search(text_feature, k=10)
for path, similarity in text_results:
    print(f"{path} (相似度: {similarity}%)")
```
