# ImgSearch

[🇬🇧 English](#imgsearch) ⇌ [🇨🇳 中文](#imgsearch-图像搜索引擎)

ImgSearch is a lightweight image search engine that supports image-to-image and text-to-image searches. Built on [TinyCLIP](https://github.com/wkcn/TinyCLIP) and [HNSWlib](https://github.com/nmslib/hnswlib), it's fast and resource-efficient, running on devices with just 2GB of RAM. Use it standalone or integrate it as a Python library.

## Features

- [x] Search by image: Upload a query image to find similar ones quickly
- [x] Search by text: Find images matching natural language descriptions
- [x] Image similarity comparison: Compute similarity scores (0-100%) between two images
- [x] Batch image addition: Add single files or folders (recursive), skipping duplicates automatically
- [x] Multi-database support: Create and manage multiple independent image libraries
- [x] Similarity threshold filtering: Filter search results by minimum similarity (e.g., ≥80%)

## Installation

### Standard Installation

ImgSearch relies on TinyCLIP models (built on OpenCLIP), which require PyTorch. The standard installation automatically pulls in the CUDA version of PyTorch. This is recommended for users with NVIDIA GPUs to speed up image inference.

```shell
pip install imgsearch
```

### CPU Environment

For non-CUDA environments, install the CPU version of PyTorch first, then install ImgSearch:

```shell
# Install CPU version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ImgSearch
pip install imgsearch
```

If using uv as your Python package manager:

```shell
uv pip install --torch-backend cpu imgsearch
```

**Note**: ImgSearch's TinyCLIP models work on both CPU and GPU. GPU users can skip the CPU-specific steps and use the standard PyTorch installation for faster performance.

## Quick Start

### 1. Service Management

ImgSearch follows a client-server architecture: the server handles indexing and search tasks, while the client manages user requests. Before using ImgSearch, start the server process.

The server supports Unix domain sockets (default for local, efficient connections) or TCP binding. The default Unix socket is at `~/.isearch/isearch.sock`.

#### Basic Usage

The default model is ViT-45LY (TinyCLIP-auto-ViT-45M-32-Text-18M-LAIONYFCC400M). See [Model Selection Guide](#model-selection-guide) for available options.

##### i. Start Service

```shell
# Start ImgSearch server with default settings
isearch service start
```

###### Optional parameters for start command

- `-B BIND`: Server binding, either `UDS` path or `IP:PORT` (e.g., `-B /path/to/isearch.sock` or `-B 127.0.0.1:5000`)
- `-b BASE_DIR`: Index database directory (default: `~/.isearch/`)
- `-m MODEL_KEY`: Model name (see [Model Selection Guide](#model-selection-guide))
- `-L LOG_LEVEL`: Log level (`debug`, `info`, `warning`, `error`, `critical`; default: `info`)

##### ii. Stop Service

```shell
isearch service stop
```

##### iii. Check Status

```shell
isearch service status
```

Example output:
```
ImgSearch service is running
* PID: 12345
* MEM: 256.3 MB
```

#### Running as a System Service

ImgSearch can run as a background system service that starts and stops with your system. On Linux, it uses `systemd`; on macOS, `launchd`. The tool auto-detects your environment—no manual configuration needed.

##### i. Set Up Service

The `setup` command creates and starts the ImgSearch system service. It accepts the same optional parameters as the [`start` command](#optional-parameters-for-start-command).

**⚠️ Note**: This downloads model files to the cache directory `~/.cache/clip`. Most models range from 100 MB to 230 MB—ensure you have enough disk space.

```shell
isearch service setup
```

##### ii. Remove Service

The `remove` command stops and uninstalls the ImgSearch system service. **It does not delete database files.**

```shell
isearch service remove
```

### 2. Add Images to Index

The `add` command indexes images in the specified database. It supports formats like jpg, jpeg, png, bmp, and webp. Folders are scanned recursively, with duplicates automatically filtered (based on labels).

```shell
# Add single files or folders to the default database
isearch add ./images/photo1.jpg ./pictures/

# Use filename as label (default: absolute path)
isearch add -l name ./images/

# Specify database and binding
isearch add -d my_gallery ./photos/ -B ./isearch.sock
```

The `add` command extracts 512-dimensional feature vectors from images using TinyCLIP and stores them in an HNSW index.

### 3. Search Images

The `search` subcommand handles image searches, but for ease of use, it's the default—omit it if your arguments don't match other commands.

Search syntax: `isearch [search] QUERY` (`[search]` optional).

#### Search by Image

```shell
# Search similar images, return top 10 results (similarity ≥0%)
isearch ./query.jpg
# Equivalent: isearch search ./query.jpg

# Set minimum similarity threshold and result count
isearch -n 5 -m 80 ./query.jpg
# Equivalent: isearch search -n 5 -m 80 ./query.jpg

# Automatically open result images
isearch -o ./query.jpg
# Equivalent: isearch search -o ./query.jpg
```

#### Search by Text

```shell
# Search for "red flower" related images
isearch "red flower"
# Equivalent: isearch search "red flower"

# Specify count and threshold
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

#### View Database Info

```shell
isearch db --info
```

Example output:
```
Database "default"
* Base: /home/user/.isearch
* Size: 140915
* Capacity: 150000
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

**⚠️ Warning**: This operation is irreversible and deletes all index data.

### 5. Compare Two Images

```shell
isearch cmp ./img1.jpg ./img2.png
```

Output:
```
Similarity between images: 87.5%
```

## As a Python Module

ImgSearch can be used as a Python module in other projects.

```python
from imgsearch.client import Client

# Create client (connects to local service by default)
cli = Client(db_name='default', bind='~/.isearch/isearch.sock')

# Add images (returns count)
image_paths = ['./img1.jpg', './img2.png', './folder/']
n_added = cli.add_images(image_paths, label_type='path')  # or 'name'
print(f'Added {n_added} images for processing')

# Search by image (returns [(path, similarity%), ...] or None)
results = cli.search('./query.jpg', num=5, similarity=80)
if results:
    for path, sim in results:
        print(f"{path} (similarity: {sim}%)")
else:
    print('No matching results or search queue full')

# Search by text
results = cli.search('red apple', num=10, similarity=0)
for path, sim in results:
    print(f"{path} (similarity: {sim}%)")

# Compare similarity (returns 0-100 float)
similarity = cli.compare_images('./img1.jpg', './img2.jpg')
print(f'Similarity: {similarity}%')

# Database operations
dbs = cli.list_dbs()
print(f'Available databases: {dbs}')

info = cli.get_db_info()
print(f'Database information: {info}')

# Clear database (returns True/False)
cleared = cli.clear_db()
print(f'Clear success: {cleared}')
```

**⚠️ Note**: Start the service first (`isearch service start`) for module usage, or connections will fail.

## Model Selection Guide

ImgSearch supports various TinyCLIP models, with the default `ViT-45LY` offering a good balance of speed, accuracy, and resource use for most cases.

| Model    | ImageNet-1K Acc@1 (%) | MACs (G) | Throughput (pairs/s) | Recommended Scenarios                                  |
|----------|-----------------------|----------|----------------------|--------------------------------------------------------|
| ViT-8Y   | 41.1                  | 2.0      | 4,150                | Lowest resource use, fast, but slightly lower accuracy |
| RN-19L   | 56.4                  | 4.4      | 3,024                |                                                        |
| ViT-22L  | 53.7                  | 1.9      | 5,504                | Fastest option, ideal for speed-critical setups        |
| RN-30L   | 59.1                  | 6.9      | 1,811                |                                                        |
| ViT-39Y  | 63.5                  | 9.5      | 1,469                | High accuracy with moderate resources, but slower      |
| ViT-40L  | 59.8                  | 3.5      | 4,641                |                                                        |
| ViT-45L  | 61.4                  | 3.7      | 3,682                |                                                        |
| ViT-45LY | 62.7                  | 1.9      | 3,685                | **Default model**, balances speed and accuracy         |
| ViT-61L  | 62.4                  | 5.3      | 3,191                |                                                        |
| ViT-63L  | 63.9                  | 5.6      | 2,905                |                                                        |
| ViT-63LY | 64.5                  | 5.6      | 2,909                | Highest accuracy                                       |

Data source: [TinyCLIP Model Zoo](https://github.com/wkcn/TinyCLIP?tab=readme-ov-file#model-zoo). Use smaller models like ViT-8Y on low-end devices; opt for larger ones like ViT-63LY for precision-focused tasks.

To switch models, restart the service:

```shell
# Restart ImgSearch service
isearch service stop
isearch service start -m NEW_MODEL_KEY
```

---

# ImgSearch 图像搜索引擎

[🇨🇳 中文](#imgsearch-图像搜索引擎) ⇌ [🇬🇧 English](#imgsearch)

ImgSearch 是一款轻量级图片搜索引擎，支持以图搜图和文字描述搜图。基于 [TinyCLIP](https://github.com/wkcn/TinyCLIP) 和 [HNSWlib](https://github.com/nmslib/hnswlib) 构建，速度快、资源占用低，可在 2GB 内存设备上运行。可作为独立搜索引擎使用，或作为 Python 库集成到其他系统。

## 特性

- [x] 以图搜图：上传查询图片，快速找到相似图像
- [x] 文字搜图：通过自然语言描述搜索相关图片
- [x] 图像相似度比较：计算两张图片的相似度分数（0-100%）
- [x] 批量添加图片：支持单个文件或文件夹（递归添加），自动跳过重复
- [x] 多数据库支持：可创建和管理多个独立图片库
- [x] 相似度阈值过滤：搜索结果可设置最小相似度（如 ≥80%）

## 安装

### 标准安装

ImgSearch 使用的 TinyCLIP 模型（基于 OpenCLIP）依赖 PyTorch。标准安装时，pip 会自动安装 CUDA 版的 PyTorch。建议使用 NVIDIA 显卡的用户选择这种方式以加速图像推理过程。

```shell
pip install imgsearch
```

### CPU 环境安装

对于非 CUDA 环境的用户，建议先安装 CPU 版 PyTorch，然后再安装 imgsearch：

```shell
# 安装 CPU 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装 ImgSearch
pip install imgsearch
```

使用 uv 作为 Python 包管理工具的可以这样安装：

```shell
uv pip install --torch-backend cpu imgsearch
```

## 使用方法

ImgSearch 整体为 C-S 架构，服务端处理索引和搜索，客户端处理用户请求。使用前需先开启他的服务端程序。

### 1. 服务管理

ImgSearch 服务端支持 Unix 域套接字或 TCP 绑定。默认使用 Unix 域套接字运行在本地，稳定且高效。默认 UDS 位置为 `~/.isearch/isearch.sock`。

#### 基本用法

##### i. 启动服务

默认模型为 ViT-45LY（`TinyCLIP-auto-ViT-45M-32-Text-18M-LAIONYFCC400M`），可用模型列表详见 [模型选择指南](#模型选择指南)。

```shell
# 使用默认参数启动 Img Search 服务
isearch service start
```

###### start 命令可选参数

- `-B BIND`:       服务端绑定方式，可选 `UDS` 或 `IP:PORT` 两种格式。如：`-B /path/to/isearch.sock` 或 `-B 127.0.0.1:5000`
- `-b BASE_DIR`:   索引数据库目录，默认为 `~/.isearch/`
- `-m MODEL_KEY`:  模型名称，可选项详见 [模型选择指南](#模型选择指南)
- `-L LOG_LEVEL`:  日志级别，可选 `debug`、`info`、`warning`、`error`、`critical`。

##### ii. 停止服务

```shell
isearch service stop
```

##### iii. 查看状态

```shell
isearch service status

# 输出：
iSearch service is running
* PID: 12345
* MEM: 256.3 MB
```

#### 将 ImgSearch 作为系统服务在后台运行

ImgSearch 可以作为系统服务在后台运行，并可以随系统自动启动和停止。在 Linux 上，ImgSearch 使用 `systemd` 来管理守护进程，在 macOS 上则使用 `launchd`。ImgSearch 会自动识别运行环境，无需手动指定服务管理程序。

##### i. 设置服务

`setup` 命令用于创建和启动 ImgSearch 的系统服务。它的可选参数与 [`start` 命令](#start-命令可选参数) 一样。

**⚠️ 注意**：此操作会下载模型文件到缓存目录 `~/.cache/clip`，大部分模型的大小在 100 MB 至 230 MB 之间，请确保磁盘空间充足。

```shell
isearch service setup
```

##### ii. 移除服务

`remove` 命令用于停止和卸载 ImgSearch 的系统服务，但 **不会删除** 数据库文件。

```shell
isearch service remove
```

### 2. 添加图片到索引

`add` 命令用于将图片添加到指定数据库。支持 jpg, jpeg, png, bmp, webp 等格式。文件夹会递归添加所有图片，自动过滤重复项（基于 Label）。

```shell
# 添加单个文件或文件夹到默认数据库
isearch add ./images/photo1.jpg ./pictures/

# 使用文件名作为标签（默认：绝对路径）
isearch add -l name ./images/

# 指定数据库和绑定
isearch add -d my_gallery ./photos/ -B ./isearch.sock
```

`add` 命令会通过 TinyCLIP 提取图片的特征（512 维向量），并存储在 HNSWlib 索引中。

### 3. 搜索图片

搜索图片使用 `search` 子命令，为了操作方便，isearch 已将它设为默认子命令，使用时可以省略。

搜图语法：`isearch [search] QUERY`（`[search]` 可选）。

#### 以图搜图

```shell
# 搜索相似图片，默认返回前 10 个结果
isearch ./query.jpg

# 等价于：
isearch search ./query.jpg

# 设置最小相似度阈值和结果数量
isearch -n 5 -m 80 ./query.jpg

# 自动打开结果图片
isearch -o ./query.jpg
```

#### 以关键字搜图

```shell
# 搜索 "red flower" 相关图片
isearch "red flower"

# 指定数量和阈值
isearch -n 3 -m 70 "sports car"
```

搜索结果按相似度降序排列，显示路径和相似度。示例：

```
Searching sports car...
Found 3 similar images (similarity ≥ 70%):
 1. /path/to/img1.jpg	92.3%
 2. /path/to/img2.png	85.1%
 3. /path/to/img3.jpg	78.4%
```

### 4. 数据库管理

#### 查看数据库信息

```shell
isearch db --info
```

输出示例：
```
Database "default"
* Base: /home/user/.isearch
* Size: 140915
* Capacity: 150000
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

**⚠️ 警告**：此操作不可逆，会删除所有索引数据。

### 5. 比较两张图片

```shell
isearch cmp ./img1.jpg ./img2.png
```

输出：
```
Similarity between images: 87.5%
```

## 作为 Python 模块使用

ImgSearch 可作为 Python 模块导入到其他项目中。

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

**⚠️ 注意**：模块使用需先启动服务（`isearch service start`），否则连接失败。

## 模型选择指南

ImgSearch 支持多种 TinyCLIP 模型，默认的 `ViT-45LY` 平衡了速度、准确率和资源占用，适用于大多数场景。

| 模型     | ImageNet-1K Acc@1 (%) | MACs (G) | Throughput (pairs/s) | 推荐场景                           |
|----------|-----------------------|----------|----------------------|------------------------------------|
| ViT-8Y   | 41.1                  | 2.0      | 4,150                | 资源消耗最低，速度快，准确度略低   |
| RN-19L   | 56.4                  | 4.4      | 3,024                |                                    |
| ViT-22L  | 53.7                  | 1.9      | 5,504                | 速度最快，适合对速度要求高的场景   |
| RN-30L   | 59.1                  | 6.9      | 1,811                |                                    |
| ViT-39Y  | 63.5                  | 9.5      | 1,469                | 准确度高，资源消耗中等，但速度较慢 |
| ViT-40L  | 59.8                  | 3.5      | 4,641                |                                    |
| ViT-45L  | 61.4                  | 3.7      | 3,682                |                                    |
| ViT-45LY | 62.7                  | 1.9      | 3,685                | **默认模型**，速度与精度兼备       |
| ViT-61L  | 62.4                  | 5.3      | 3,191                |                                    |
| ViT-63L  | 63.9                  | 5.6      | 2,905                |                                    |
| ViT-63LY | 64.5                  | 5.6      | 2,909                | 准确度最高                         |

数据来源：[TinyCLIP 模型库](https://github.com/wkcn/TinyCLIP?tab=readme-ov-file#model-zoo)。选择小模型（如 ViT-8Y）用于低端设备；大模型（如 ViT-63LY）用于高精度需求。

切换模型需重启服务：

```shell
# 重启 isearch 服务
isearch service stop
isearch service start -m NEW_MODEL_KEY
