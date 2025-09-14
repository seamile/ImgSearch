# ImgSearch 图片搜索引擎

[🇨🇳 中文](#imgsearch-图片搜索引擎) ⇌ [🇬🇧 English](#imgsearch)

ImgSearch 是一款轻量级图片搜索引擎，支持以图搜图和文字描述搜图。基于 TinyCLIP（OpenCLIP 兼容）和 HNSWlib 构建，速度快、资源占用低，可在 2GB 内存设备上运行。可作为独立搜索引擎使用，或作为 Python 库集成到其他系统。

## 特性

- [x] 以图搜图：上传查询图片，快速找到相似图像
- [x] 文字搜图：通过自然语言描述搜索相关图片
- [x] 图像相似度比较：计算两张图片的相似度分数（0-100%）
- [x] 批量添加图片：支持单个文件或文件夹（递归添加），自动跳过重复
- [x] 多数据库支持：可创建和管理多个独立图片库
- [x] 相似度阈值过滤：搜索结果可设置最小相似度（如 ≥80%）

## 安装

### 标准安装

```shell
pip install imgsearch
```

### CPU 环境安装

在纯 CPU 环境（如无 NVIDIA GPU 的 macOS 或服务器）下，先安装 CPU 版 PyTorch：

```shell
# macOS Intel/Apple Silicon 或 Linux CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 然后安装 ImgSearch
pip install imgsearch
```

**注意**：ImgSearch 使用 TinyCLIP 模型（基于 OpenCLIP），兼容 CPU/GPU。GPU 用户可直接安装标准 PyTorch（`pip install torch torchvision`）以加速推理。

## 快速开始

### 1. 服务管理

ImgSearch 使用后台服务处理索引和搜索。服务支持 Unix 域套接字（默认，本地高效）或 TCP 绑定。

#### 启动服务

```shell
# 默认：使用 ~/.isearch/isearch.sock，本地连接
isearch service start

# 指定模型和日志级别
isearch service start -m ViT-45LY -L info

# TCP 绑定（例如：监听 127.0.0.1:8080）
isearch service start -B 127.0.0.1:8080
```

可用模型键：`ViT-8Y`, `RN-19L`, `ViT-22L`, `RN-30L`, `ViT-39Y`, `ViT-40L`, `ViT-45L`, `ViT-45LY` (默认), `ViT-61L`, `ViT-63L`, `ViT-63LY`。详见 [模型选择指南](#模型选择指南)。

#### 停止服务

```shell
isearch service stop
```

#### 查看状态

```shell
isearch service status
```

输出示例：
```
iSearch service is running
* PID: 12345
* MEM: 256.3 MB
```

### 2. 添加图片到索引

将图片添加到指定数据库。支持 jpg, jpeg, png, bmp, webp 等格式。文件夹会递归添加所有图片，自动过滤重复（基于标签）。

```shell
# 添加单个文件或文件夹到默认数据库
isearch add ./images/photo1.jpg ./dataset/

# 使用文件名作为标签（默认：绝对路径）
isearch add -l name ./images/

# 指定数据库和绑定
isearch add -d my_gallery ./photos/ -B ./isearch.sock
```

添加后，图片会被转换为 384x384 并提取 TinyCLIP 特征（512 维向量），存储在 HNSW 索引中。

### 3. 搜索图片

搜索图片使用 `search` 子命令，为了操作方便，isearch 已将它设为默认子命令，使用时可以省略。

搜图语法：`isearch [search] QUERY`（`[search]` 可选）。

#### 以图搜图

```shell
# 搜索相似图片，返回前 10 个结果（相似度 ≥0%）
isearch ./query.jpg
# 等价于：isearch search ./query.jpg

# 设置最小相似度阈值和结果数量
isearch -n 5 -m 80 ./query.jpg
# 等价于：isearch search -n 5 -m 80 ./query.jpg

# 自动打开结果图片
isearch -o ./query.jpg
# 等价于：isearch search -o ./query.jpg
```

#### 以文字搜图

```shell
# 搜索 "red flower" 相关图片
isearch "red flower"
# 等价于：isearch search "red flower"

# 指定数量和阈值
isearch -n 3 -m 70 "sunset beach"
# 等价于：isearch search -n 3 -m 70 "sunset beach"
```

结果按相似度降序排列，显示路径和百分比分数。示例输出：
```
Searching red flower...
Found 5 similar images (similarity ≥ 70.0%):
 1. /path/to/img1.jpg	92.3%
 2. /path/to/img2.png	85.1%
 3. /path/to/img3.jpg	78.4%
```

### 4. 数据库管理

管理单个或多个数据库（每个数据库独立索引）。

#### 查看数据库信息

```shell
isearch db --info
```

输出示例：
```
Database "default"
* Base: /Users/user/.isearch/default
* Size: 1245
* Capacity: 10000
```

#### 列出所有数据库

```shell
isearch db --list
```

输出：
```
Available databases:
* default
* my_gallery
* test_db
```

#### 清空数据库

```shell
# 确认后清空
isearch db --clear -d my_db
```

**警告**：此操作不可逆，会删除所有索引数据。

### 5. 比较两张图片

```shell
isearch cmp ./img1.jpg ./img2.png
```

输出：
```
Similarity between images: 87.5%
```

### 6. 命令行参数

全局参数（适用于所有子命令）：
- `-d DB_NAME`：指定数据库名称（默认：`default`）
- `-B BIND`：服务绑定地址（默认：`~/.isearch/isearch.sock`；TCP 格式：`host:port`）
- `-v, --version`：显示版本（当前：0.2.0）

子命令特定参数见上文。

## 作为 Python 模块使用

ImgSearch 可导入为库，直接操作服务或独立使用（但推荐服务模式以支持并发）。

```python
from imgsearch.client import Client

# 创建客户端（默认连接本地服务）
cli = Client(db_name='default', bind='~/.isearch/isearch.sock')

# 添加图片（返回添加数量）
image_paths = ['./img1.jpg', './img2.png', './folder/']
n_added = cli.add_images(image_paths, label_type='path')  # 或 'name'
print(f'已添加 {n_added} 张图片进行处理')

# 以图搜图（返回 [(路径, 相似度%), ...] 或 None）
results = cli.search('./query.jpg', num=5, similarity=80)
if results:
    for path, sim in results:
        print(f"{path} (相似度: {sim}%)")
else:
    print('无匹配结果或搜索队列满')

# 以文字搜图
results = cli.search('red apple', num=10, similarity=0)
for path, sim in results:
    print(f"{path} (相似度: {sim}%)")

# 比较相似度（返回 0-100 float）
similarity = cli.compare_images('./img1.jpg', './img2.jpg')
print(f'相似度: {similarity}%')

# 数据库操作
dbs = cli.list_dbs()
print(f'可用数据库: {dbs}')

info = cli.get_db_info()
print(f'数据库信息: {info}')

# 清空数据库（返回 True/False）
cleared = cli.clear_db()
print(f'清空成功: {cleared}')
```

**注意**：模块使用需先启动服务（`isearch service start`），否则连接失败。

## 模型选择指南

ImgSearch 支持多种 TinyCLIP 模型变体，平衡速度、准确率和资源。默认 `ViT-45LY` 适用于大多数场景。

| 模型键   | ImageNet-1K Acc@1 (%) | MACs (G) | Throughput (pairs/s) | 推荐场景                           |
|----------|-----------------------|----------|----------------------|------------------------------------|
| ViT-8Y   | 41.1                  | 2.0      | 4,150                | 资源消耗最低，速度快，准确度略低   |
| RN-19L   | 56.4                  | 4.4      | 3,024                |                                    |
| ViT-22L  | 53.7                  | 1.9      | 5,504                | 速度最快，适合对速度要求高的场景   |
| RN-30L   | 59.1                  | 6.9      | 1,811                |                                    |
| ViT-39Y  | 63.5                  | 9.5      | 1,469                | 准确度高，资源消耗中等，但速度较慢 |
| ViT-40L  | 59.8                  | 3.5      | 4,641                |                                    |
| ViT-45L  | 61.4                  | 3.7      | 3,682                |                                    |
| ViT-45LY | 62.7                  | 1.9      | 3,685                | **默认：最佳均衡**，速度与精度兼备 |
| ViT-61L  | 62.4                  | 5.3      | 3,191                |                                    |
| ViT-63L  | 63.9                  | 5.6      | 2,905                |                                    |
| ViT-63LY | 64.5                  | 5.6      | 2,909                | 准确度最高                         |

数据来源：TinyCLIP 模型库。选择小模型（如 ViT-8Y）用于低端设备；大模型（如 ViT-63LY）用于高精度需求。切换模型需重启服务：`isearch service start -m MODEL_KEY`。

## 目录结构

```
src/imgsearch/
├── __init__.py          # 包初始化
├── __main__.py          # CLI 入口点
├── client.py            # 客户端 API 和解析器
├── server.py            # 服务端逻辑（Pyro5 RPC、信号处理）
├── clip.py              # 特征提取（TinyCLIP 集成）
├── storage.py           # 向量数据库（HNSWlib + bidict）
├── utils.py             # 工具函数（图像处理、日志、颜色输出）
└── consts.py            # 常量定义（模型、路径、配置）
```

测试目录：`src/test/`（包含单元测试，使用 pytest）。

## 许可证

MIT License

---

# ImgSearch

[🇬🇧 English](#imgsearch) ⇌ [🇨🇳 中文](#imgsearch-图片搜索引擎)

ImgSearch is a lightweight image search engine supporting image-to-image and text-to-image search. Built on TinyCLIP (OpenCLIP-compatible) and HNSWlib, it's fast and resource-efficient, running on 2GB RAM devices. Use standalone or integrate as a Python library.

## Features

- [x] Image-to-image search: Find similar images from query photo
- [x] Text-to-image search: Search by natural language descriptions
- [x] Image similarity comparison: Compute score (0-100%) between two images
- [x] Batch image addition: Single files or folders (recursive), auto-deduplicate
- [x] Multi-database support: Manage multiple independent image collections
- [x] Similarity threshold filtering: Filter results by min similarity (e.g., ≥80%)

## Installation

### Standard Installation

```shell
pip install imgsearch
```

### CPU Environment

For CPU-only setups (e.g., no NVIDIA GPU on macOS/server):

```shell
# Install CPU PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install ImgSearch
pip install imgsearch
```

**Note**: ImgSearch uses TinyCLIP models (OpenCLIP-based), compatible with CPU/GPU. GPU users: Install standard PyTorch (`pip install torch torchvision`) for acceleration.

## Quick Start

### 1. Service Management

ImgSearch runs as a background service for indexing/search. Supports Unix domain sockets (default, local-efficient) or TCP binding.

#### Start Service

```shell
# Default: Use ~/.isearch/isearch.sock for local connection
isearch service start

# Specify model and log level
isearch service start -m ViT-45LY -L info

# TCP binding (e.g., listen on 127.0.0.1:8080)
isearch service start -B 127.0.0.1:8080
```

Available model keys: `ViT-8Y`, `RN-19L`, `ViT-22L`, `RN-30L`, `ViT-39Y`, `ViT-40L`, `ViT-45L`, `ViT-45LY` (default), `ViT-61L`, `ViT-63L`, `ViT-63LY`. See [Model Selection Guide](#model-selection-guide).

#### Stop Service

```shell
isearch service stop
```

#### Check Status

```shell
isearch service status
```

Example output:
```
iSearch service is running
* PID: 12345
* MEM: 256.3 MB
```

### 2. Add Images to Index

Add to specified database. Supports jpg, jpeg, png, bmp, webp. Folders recursively add all images, auto-skip duplicates (by label).

```shell
# Add files/folders to default DB
isearch add ./images/photo1.jpg ./dataset/

# Use filename as label (default: absolute path)
isearch add -l name ./images/

# Specify DB and bind
isearch add -d my_gallery ./photos/ -B ./isearch.sock
```

Images converted to 384x384, extract TinyCLIP features (512-dim vectors), stored in HNSW index.

### 3. Search Images

Search images uses the `search` subcommand, but for operational convenience, iSearch has set it as the default subcommand, which can be omitted during use. If no subcommand is specified and the arguments don't match other commands, it will be automatically treated as a search.

Search images syntax: `isearch [search] QUERY` (`[search]` optional).

#### Image-to-Image Search

```shell
# Search similar images, top 10 (similarity ≥0%)
isearch ./query.jpg
# Equivalent: isearch search ./query.jpg

# Set min similarity and num results
isearch -n 5 -m 80 ./query.jpg
# Equivalent: isearch search -n 5 -m 80 ./query.jpg

# Auto-open results
isearch -o ./query.jpg
# Equivalent: isearch search -o ./query.jpg
```

#### Text-to-Image Search

```shell
# Search "red flower" related images
isearch "red flower"
# Equivalent: isearch search "red flower"

# Specify num and threshold
isearch -n 3 -m 70 "sunset beach"
# Equivalent: isearch search -n 3 -m 70 "sunset beach"
```

Results sorted by similarity descending, show path and percentage. Example:
```
Searching red flower...
Found 5 similar images (similarity ≥ 70.0%):
 1. /path/to/img1.jpg	92.3%
 2. /path/to/img2.png	85.1%
 3. /path/to/img3.jpg	78.4%
```

### 4. Database Management

Manage single/multiple databases (each independent index).

#### View Database Info

```shell
isearch db --info
```

Example:
```
Database "default"
* Base: /Users/user/.isearch/default
* Size: 1245
* Capacity: 10000
```

#### List All Databases

```shell
isearch db --list
```

Output:
```
Available databases:
* default
* my_gallery
* test_db
```

#### Clear Database

```shell
# Confirm then clear
isearch db --clear -d my_db
```

**Warning**: Irreversible, deletes all index data.

### 5. Compare Two Images

```shell
isearch cmp ./img1.jpg ./img2.png
```

Output:
```
Similarity between images: 87.5%
```

### 6. Command-Line Arguments

Global args (all subcommands):
- `-d DB_NAME`: Database name (default: `default`)
- `-B BIND`: Service bind (default: `~/.isearch/isearch.sock`; TCP: `host:port`)
- `-v, --version`: Show version

Subcommand-specific: See above.

## As a Python Module

Import for library use, operates on service or standalone (service recommended for concurrency).

```python
from imgsearch.client import Client

# Create client (default local service)
cli = Client(db_name='default', bind='~/.isearch/isearch.sock')

# Add images (returns count)
image_paths = ['./img1.jpg', './img2.png', './folder/']
n_added = cli.add_images(image_paths, label_type='path')  # or 'name'
print(f'Added {n_added} images for processing')

# Image-to-image search (returns [(path, similarity%), ...] or None)
results = cli.search('./query.jpg', num=5, similarity=80)
if results:
    for path, sim in results:
        print(f"{path} (similarity: {sim}%)")
else:
    print('No matches or search queue full')

# Text-to-image search
results = cli.search('red apple', num=10, similarity=0)
for path, sim in results:
    print(f"{path} (similarity: {sim}%)")

# Compare similarity (returns 0-100 float)
similarity = cli.compare_images('./img1.jpg', './img2.jpg')
print(f'Similarity: {similarity}%')

# Database ops
dbs = cli.list_dbs()
print(f'Available databases: {dbs}')

info = cli.get_db_info()
print(f'DB info: {info}')

# Clear DB (returns True/False)
cleared = cli.clear_db()
print(f'Cleared: {cleared}')
```

**Note**: Requires running service (`isearch service start`), else connection fails.

## Model Selection Guide

ImgSearch supports various TinyCLIP model variants, balancing speed, accuracy, resources. Default `ViT-45LY` for most cases.

| Model Key | ImageNet-1K Acc@1 (%) | MACs (G) | Throughput (pairs/s) | Recommended Scenarios                                                 |
|-----------|-----------------------|----------|----------------------|-----------------------------------------------------------------------|
| ViT-8Y    | 41.1                  | 2.0      | 4,150                | Lowest resource usage, fast performance, with slightly lower accuracy |
| RN-19L    | 56.4                  | 4.4      | 3,024                |                                                                       |
| ViT-22L   | 53.7                  | 1.9      | 5,504                | Fastest speed, ideal for scenarios with high speed requirements       |
| RN-30L    | 59.1                  | 6.9      | 1,811                |                                                                       |
| ViT-39Y   | 63.5                  | 9.5      | 1,469                | High accuracy, moderate resource consumption, but slower speed        |
| ViT-40L   | 59.8                  | 3.5      | 4,641                |                                                                       |
| ViT-45L   | 61.4                  | 3.7      | 3,682                |                                                                       |
| ViT-45LY  | 62.7                  | 1.9      | 3,685                | **Default: Best balance** of speed and accuracy                       |
| ViT-61L   | 62.4                  | 5.3      | 3,191                |                                                                       |
| ViT-63L   | 63.9                  | 5.6      | 2,905                |                                                                       |
| ViT-63LY  | 64.5                  | 5.6      | 2,909                | Highest accuracy                                                      |

Data source: TinyCLIP model library. Choose small models (e.g., ViT-8Y) for low-end devices; large models (e.g., ViT-63LY) for high-precision needs. To switch models, restart the service: `isearch service start -m MODEL_KEY`.

## Directory Structure

```
src/imgsearch/
├── __init__.py          # Package init
├── __main__.py          # CLI entrypoint
├── client.py            # Client API and parser
├── server.py            # Server logic (Pyro5 RPC, signals)
├── clip.py              # Feature extraction (TinyCLIP integration)
├── storage.py           # Vector DB (HNSWlib + bidict)
├── utils.py             # Utilities (image proc, logging, colors)
└── consts.py            # Constants (models, paths, config)
```

Tests: `src/test/` (unit tests with pytest).

## License

MIT License
