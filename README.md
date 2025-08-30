# Pixa 搜图

[🇨🇳 中文](#Pixa-搜图) ⇌ [🇬🇧 English](#Pixa)

Pixa 是一款轻量级的图片搜索引擎，可以用来以图搜图，也可以通过文字描述搜索图片。资源占用低，适合部署在性能有限的机器上。可以依靠显卡提升性能，也可以完全以纯 CPU 模式运行。支持命令行直接使用，也可作为 Python 模块集成到其他系统。

## 特性

- [x] 支持以图搜图、以关键字搜图
- [x] 支持对比两张图片的相似度
- [x] 支持批量添加图片（文件或文件夹）
- [x] 支持指定 CLIP 模型

## 安装

```shell
pip install pixa
```

## 快速开始

### 1. 服务管理

#### 启动服务

```shell
px -s start
```

#### 停止服务

```shell
px -s stop
```

#### 查看状态

```shell
px -s status
```

### 2. 添加图片到图像索引

将指定路径的图片添加到数据库中。目标可为单张图片或文件夹（递归添加所有图片）。支持 jpg、jpeg、png、bmp、webp 等常见格式。

```shell
px -a ./foo/img1.png ./bar/img2.jpg ./path/to/images_dir/
```

### 3. 搜索图片

通过样本图片或描述信息搜索图片。可通过 `-n` 参数指定返回结果数量，结果按相似度降序排列。

#### 以图搜图

```shell
px /path/to/query_image.jpg
```

#### 以关键词搜图

```shell
px -n 3 "red flower"
```

### 4. 其他功能

#### 查看数据库信息

```shell
px -i
```

#### 比较两张图片的相似度

```shell
px -c ./foo/img1.png ./bar/img2.jpg
```

#### 清空数据库

```shell
px -C
```

### 5. 可选参数列表

- `-d DB_DIR`   指定数据库目录路径（默认: ~/.pixa）
- `-l {path,name}` 标签命名方式：path=绝对路径，name=文件名
- `-m MODEL`    CLIP 模型名称（如 tinyclip-vit-large）
- `-n NUM`      搜索结果数量（默认: 10）

### 6. 作为模块导入

Pixa 也可作为 Python 模块集成：

```python
from pixa.client import Client

# 创建客户端
cli = Client()

# 添加图片到索引
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
n_added = cli.add_images(image_paths)
print(f'添加了 {n_added} 张图片')

# 以图搜图
for path, similarity in cli.search("/path/to/query.jpg"):
    print(f"{path} (相似度: {similarity}%)")

# 以关键词搜图
for path, similarity in cli.search("red flower"):
    print(f"{path} (相似度: {similarity}%)")

# 对比相似度
similarity = cli.compare("/path/to/image1.jpg", "/path/to/image2.jpg")
print(f'相似度: {similarity}%')
```

## 目录结构

```
pixa/
├ __main__.py      # 命令行入口
├ client.py        # 客户端 API
├ server.py        # 服务端主逻辑
├ clip.py          # 特征提取
├ storage.py       # 数据存储与索引
├ utils.py         # 工具函数
└ consts.py        # 常量定义
```

## 依赖

- Python >=3.8
- torch
- hnswlib
- Pillow

## 许可证

MIT License

---

# Pixa

[🇬🇧 English](#Pixa) ⇌ [🇨🇳 中文](#Pixa-搜图)

Pixa is a lightweight image search engine that supports both image-to-image and text-to-image search. It is resource-efficient and can be deployed on low-end machines. You can use it directly from the command line or integrate it as a Python module in other systems.

## Features

- [x] Search by image or text
- [x] Compare similarity between two images
- [x] Batch add images (files or folders)
- [x] Customizable CLIP model

## Installation

```shell
pip install pixa
```

## Quick Start

### 1. Service Management

#### Start Service

```shell
px -s start
```

#### Stop Service

```shell
px -s stop
```

#### Check Status

```shell
px -s status
```

### 2. Add Images to Index

Add images to the database by specifying file or folder paths (recursively adds all images in folders). Supports jpg, jpeg, png, bmp, webp, etc.

```shell
px -a ./foo/img1.png ./bar/img2.jpg ./path/to/images_dir/
```

### 3. Search Images

Search by sample image or text description. Use `-n` to specify the number of results, sorted by similarity.

#### Image-to-Image Search

```shell
px /path/to/query_image.jpg
```

#### Text-to-Image Search

```shell
px -n 3 "red flower"
```

### 4. Other Features

#### View Database Info

```shell
px -i
```

#### Compare Two Images

```shell
px -c ./foo/img1.png ./bar/img2.jpg
```

#### Clear Database

```shell
px -C
```

### 5. Optional Arguments

- `-d DB_DIR`   Specify database directory (default: ~/.pixa)
- `-l {path,name}` Label naming: path=absolute path, name=file name
- `-m MODEL`    CLIP model name (e.g. tinyclip-vit-large)
- `-n NUM`      Number of search results (default: 10)

### 6. As a Python Module

Pixa can also be used as a Python module:

```python
from pixa.client import Client

# Create client
cli = Client()

# Add images to index
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
n_added = cli.add_images(image_paths)
print(f'Added {n_added} images')

# Image-to-image search
for path, similarity in cli.search("/path/to/query.jpg"):
    print(f"{path} (similarity: {similarity}%)")

# Text-to-image search
for path, similarity in cli.search("red flower"):
    print(f"{path} (similarity: {similarity}%)")

# Compare similarity
similarity = cli.compare("/path/to/image1.jpg", "/path/to/image2.jpg")
print(f'Similarity: {similarity}%')
```

## Directory Structure

```
pixa/
├ __main__.py      # CLI entry
├ client.py        # Client API
├ server.py        # Server logic
├ clip.py          # Feature extraction
├ storage.py       # Storage and index
├ utils.py         # Utilities
└ consts.py        # Constants
```

## Dependencies

- Python >=3.8
- torch
- hnswlib
- Pillow

## License

MIT License
