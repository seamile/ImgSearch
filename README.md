# ImgSearch - Lightweight Reverse Image Search Engine

[🇬🇧 English](#imgsearch---lightweight-reverse-image-search-engine) ⇌ [🇨🇳 中文](#imgsearch-反向图片搜索引擎)

ImgSearch is a reverse image search engine that supports search by image or text. Built on [TinyCLIP](https://github.com/wkcn/TinyCLIP) and [LanceDB](https://lancedb.com/), it stores vectors locally without a separate database server. Use it standalone or integrate it as a Python library.

## Features

- [x] Search by image: Upload a query image to find similar ones quickly
- [x] Search by text: Find images matching natural language descriptions
- [x] Image similarity comparison: Compute similarity scores (0-100%) between two images
- [x] Batch image addition: Add single files or folders (recursive), skipping duplicates automatically
- [x] Multi-database support: Create and manage multiple independent image libraries
- [x] Similarity threshold filtering: Filter search results by minimum similarity (e.g., ≥80%)

## Installation

### Full Installation (Client + Server)

To use the full functionality of ImgSearch (including client and server), you need to specify the `all` dependency group during installation. This will install all the dependency packages required to run the server, such as `PyTorch`, on the system.

```shell
pip install 'imgsearch[all]'
```

### Standard Installation (Client Only)

The standard installation includes only the client-side dependencies:

```shell
pip install imgsearch
```

**⚠️ Note**: Since the standard installation lacks PyTorch and other dependencies, you cannot run the `isearch service start` command.

### Non-CUDA environment

For users in non-CUDA environments who want to run the server, it is recommended to install the CPU version of PyTorch first, and then install ImgSearch:

```shell
# Install CPU version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ImgSearch
pip install 'imgsearch[all]'
```

If using uv as your Python package manager:

```shell
uv pip install --torch-backend cpu imgsearch
```

## Quick Start

### 1. Service Management

ImgSearch follows a client-server architecture: the server handles indexing and search tasks, while the client manages user requests. Before using ImgSearch, start the server process.

The server supports unix domain sockets (default for local, efficient connections) or TCP binding. The default unix socket is at `~/.isearch/isearch.sock`.

#### Basic Usage

The default model is ViT-45LY (`TinyCLIP-auto-ViT-45M-32-Text-18M-LAIONYFCC400M`). See [Model Selection Guide](#model-selection-guide) for available options.

##### i. Start Service

```shell
# Start ImgSearch server with default settings
isearch service start
```

###### Optional parameters for start command

- `-B BIND`: Server binding, either `UDS` path or `IP:PORT` (e.g., `-B /path/to/isearch.sock` or `-B 127.0.0.1:5000`)
- `-d DB_PATH`: LanceDB root directory (default: `~/.isearch/`)
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
iSearch service is running
 - PID  : 86269
 - MEM  : 1301.6 MB
 - Dataase : /home/seamile/.isearch
 - Model: ViT-45LY
```

#### Running as a System Service

ImgSearch can run as a background system service that starts and stops with your system. On Linux, it uses `systemd`; on macOS, `launchd`. The tool auto-detects your environment — no manual configuration needed.

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

The `add` command indexes images in the specified table. It supports formats like jpg, jpeg, png, bmp, and webp. Folders are scanned recursively. The default `--dedup-mode fast` skips filenames whose stem matches an existing label before reading or embedding them. `--dedup-mode strict` embeds every image and merges paths by vector ID, allowing repeated labels.

```shell
# Add single files or folders to the default table
isearch add ./images/photo1.jpg ./pictures/

# Embed all images and merge paths by vector ID
isearch add --dedup-mode strict ./images/

# Update labels of existing vector IDs from incoming filenames
isearch add --dedup-mode strict --update-label ./images/

# Specify table and binding
isearch add -d my_gallery ./photos/ -B ./isearch.sock
```

Fast mode compares against current labels in the selected table, not filenames from every stored path. Repeated labels in queued batches are also filtered before embedding. Skipped images do not add paths or update labels; use strict mode for different images with the same stem, replaced files, or merging new paths. `--update-label` applies only to images that reach embedding. Both modes ultimately merge by vector ID. The reported count is images accepted into the queue, not rows written. Restart the service after updating so both client and server support the label check.

The `add` command extracts feature vectors using TinyCLIP (512 or 1024 dimensions, depending on the model) and stores them in a local LanceDB table. Small tables use exact cosine search; an IVF-HNSW index is created at 10,000 items. Storage and indices are optimized after every 20 data modifications.

### 3. Search Images

The `search` subcommand handles image searches, but for ease of use, it's the default—omit it if your arguments don't match other commands.

Search command: `isearch [search] QUERY`.

#### Search by Image

```shell
# Search similar images, return top 10 results (similarity ≥0%)
isearch ./query.jpg

# Equivalent:
isearch search ./query.jpg

# Set minimum similarity threshold and result count
isearch -n 5 -t 80 ./query.jpg

# Automatically open result images (the Label when adding an image must be a path)
isearch -o ./query.jpg
```

#### Search by Text

```shell
# Search for "red flower" related images
isearch "red flower"

# Specify count and threshold
isearch -n 3 -t 70 "sports car"
```

Results sorted by similarity descending, show path and percentage. Example:
```
Searching sports car...
Found 5 similar images (similarity ≥ 70.0%):
b681d3e48ac00c09	img1	92.3%	['/path/to/img1.jpg']
1dff079456ec80b0	img2	85.1%	['/path/to/img2.png']
```

### 4. Table Management

#### List All Tables

```shell
isearch table -l
```

Output:
```
Available tables:
* default (loaded)
* gallery
* test_table
```

#### View Table Info

```shell
isearch table -i my_table
```

Example output:
```
Table "my_table"
 - DbPath: /home/seamile/.isearch
 - Count: 141610
 - Size: 313.9 MB
```

#### Delete Specific Data

```shell
isearch table my_table --delete img1 img2
isearch table my_table --delete-id b681d3e48ac00c09
```

Both deletion modes show all matching IDs, labels, and paths for confirmation. Label deletion includes every matching record.

#### Repair Paths

```shell
isearch table --prune          # default table only
isearch table --prune my_table
```

Prune removes missing paths and duplicates within a record. Paths shared by different IDs are re-embedded using the add preprocessing pipeline and associated with the matching ID, creating it if needed. Other paths and records with empty paths are retained. Failed reads/inference preserve the affected associations. Mount all relevant disks first; paths belong to the server filesystem.

#### Clear Table

```shell
# Confirm then clear
isearch table --clear my_table
```

#### Drop Table

```shell
# Confirm then drop
isearch table --drop my_table
```

**⚠️ Warning**: All delete operations are irreversible. `--clear` deletes all records in the selected table, `--drop` deletes only the selected table.

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
cli = Client(table_name='default', bind='~/.isearch/isearch.sock')

# Submit images (returns accepted queue count, not new record count)
image_paths = ['./img1.jpg', './img2.png', './folder/']
n_added = cli.add_images(image_paths, update_label=False, dedup_mode='fast')
print(f'Submitted {n_added} images for processing')

# Search by image (returns id/label/paths/score records or None)
results = cli.search('./query.jpg', num=5, similarity=80)
if results:
    for row in results or []:
        print(row["id"], row["label"], row["paths"], row["score"])
else:
    print('No matching results or search queue full')

# Search by text
results = cli.search('red apple', num=10, similarity=0)
for row in results or []:
    print(row["id"], row["label"], row["paths"], row["score"])

# Compare similarity (returns 0-100 float)
similarity = cli.compare_images('./img1.jpg', './img2.jpg')
print(f'Similarity: {similarity}%')

# Table operations
tables = cli.list_tables()
print(f'Available tables: {tables}')

info = cli.get_table_info()
print(f'Table information: {info}')

# Clear table (returns True/False)
cleared = cli.clear_table()
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

# ImgSearch 反向图片搜索引擎

[🇨🇳 中文](#imgsearch-反向图片搜索引擎) ⇌ [🇬🇧 English](#imgsearch---lightweight-reverse-image-search-engine)

ImgSearch 是一款反向图片搜索引擎，支持以图搜图和文字描述搜图。基于 [TinyCLIP](https://github.com/wkcn/TinyCLIP) 和 [LanceDB](https://lancedb.com/) 构建，向量在本地存储，无需单独部署数据库服务。可作为独立搜索引擎使用，或作为 Python 库集成到其他系统。

## 特性

- [x] 以图搜图：上传查询图片，快速找到相似图像
- [x] 文字搜图：通过自然语言描述搜索相关图片
- [x] 图像相似度比较：计算两张图片的相似度分数（0-100%）
- [x] 批量添加图片：支持单个文件或文件夹（递归添加），自动跳过重复
- [x] 多表支持：可创建和管理多个独立图片库
- [x] 相似度阈值过滤：搜索结果可设置最小相似度（如 ≥80%）

## 安装

### 完整安装（客户端 + 服务端）

要使用 ImgSearch 的完整功能（包含客户端和服务端），需要在安装时指定 `all` 依赖组，这会将服务端运行所需的 `PyTorch` 等依赖包全部安装到系统上：

```shell
pip install 'imgsearch[all]'
```

### 标准安装（仅客户端）

标准安装仅包含客户端所需的依赖：

```shell
pip install imgsearch
```

**⚠️ 注意**：由于标准安装缺少 Pytorch 等依赖项，所以无法执行 `isearch service start` 命令。

### 非 CUDA 环境

对于非 CUDA 环境的用户如果要运行服务端时，建议先安装 CPU 版 PyTorch，然后再安装 ImgSearch：

```shell
# 安装 CPU 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装 ImgSearch
pip install 'imgsearch[all]'
```

使用 uv 作为 Python 包管理工具的可以这样安装：

```shell
uv pip install --torch-backend cpu 'imgsearch[all]'
```

## 使用方法

### 1. 服务管理

ImgSearch 整体为 C-S 架构，服务端处理索引和搜索，客户端处理用户请求。使用前需先开启他的服务端程序。

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
- `-d DB_PATH`:   LanceDB 数据库根目录，默认为 `~/.isearch/`
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
 - PID  : 86269
 - MEM  : 1301.6 MB
 - DB path : /home/seamile/.isearch
 - Model: ViT-45LY
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

`add` 命令用于将图片添加到指定表。支持 jpg, jpeg, png, bmp, webp 等格式。文件夹会递归扫描，默认 `--dedup-mode fast` 在读取图片和计算向量前，按文件名 stem 查询已有 label 并跳过。`--dedup-mode strict` 对所有图片计算向量，再按向量 ID 去重并合并路径，允许 label 重名。

```shell
# 添加单个文件或文件夹到默认表
isearch add ./images/photo1.jpg ./pictures/

# 对所有图片计算向量，按向量 ID 合并路径
isearch add --dedup-mode strict ./images/

# 对已有向量 ID，使用本次文件名更新 label（默认保留原 label）
isearch add --dedup-mode strict --update-label ./images/

# 指定表和绑定
isearch add -d my_gallery ./photos/ -B ./isearch.sock
```

fast 模式只检查所选表当前的 label，不检查所有已存路径的文件名；队列中重复的 label 也会在向量化前过滤。跳过的图片不会合并路径或更新 label；同名不同图、原文件被替换、需要合并新路径时请使用 strict。`--update-label` 只作用于实际进入向量处理的图片。两种模式最终都按向量 ID 合并，输出数量为接受入队的图片数，不是成功写入的记录数。更新后需重启服务，让客户端和服务端都支持 label 检查。

`add` 命令通过 TinyCLIP 提取图片特征（按模型为 512 或 1024 维），并存储在本地 LanceDB 表中。小表使用精确 cosine 搜索，达到 10,000 条时自动创建 IVF-HNSW 索引，每 20 次数据修改执行存储和索引优化。

### 3. 搜索图片

搜索图片使用 `search` 子命令，为了操作方便，isearch 已将它设为默认子命令，使用时可以省略。

搜图方式：`isearch [search] QUERY`。

#### 以图搜图

```shell
# 搜索相似图片，默认返回前 10 个结果
isearch ./query.jpg

# 等价于：
isearch search ./query.jpg

# 设置最小相似度阈值和结果数量
isearch -n 5 -t 80 ./query.jpg

# 自动打开结果图片（添加图片时的 Label 必须是 path）
isearch -o ./query.jpg
```

#### 以关键字搜图

```shell
# 搜索 "red flower" 相关图片
isearch "red flower"

# 指定数量和阈值
isearch -n 3 -t 70 "sports car"
```

搜索结果按相似度降序排列，显示路径和相似度。示例：

```
Searching sports car...
Found 3 similar images (similarity ≥ 70%):
b681d3e48ac00c09	img1	92.3%	['/path/to/img1.jpg']
1dff079456ec80b0	img2	85.1%	['/path/to/img2.png']
```

### 4. 表管理

#### 列出所有表

```shell
isearch table -l
```

输出：
```
Available tables:
* default (loaded)
* gallery
* test_table
```

#### 查看表信息

```shell
isearch table -i my_table
```

输出示例：
```
Table "my_table"
 - DbPath: /home/seamile/.isearch
 - Count: 141610
 - Size: 313.9 MB
```

#### 删除指定数据

```shell
isearch table my_table --delete img1 img2
isearch table my_table --delete-id b681d3e48ac00c09
```

两种删除方式都会完整列出匹配记录的 ID、label 和 paths，确认后删除。按 label 删除时包含所有同名记录。

#### 修复路径

```shell
isearch table --prune          # 仅处理 default 表
isearch table --prune my_table
```

prune 移除失效路径及同 ID 内的重复路径。跨 ID 共享的路径通过与 add 相同的预处理重新计算向量，再关联到匹配的 ID，必要时新建记录。其他路径和 paths 为空的记录保留；读取或推理失败时保留相关关联。路径属于服务端文件系统，运行前请挂载相关磁盘。

#### 清空表

```shell
# 确认后清空
isearch table --clear my_table
```

#### 删除表

```shell
# 确认后删除
isearch table --drop my_table
```

**⚠️ 警告**：所有删除操作均不可逆，`--clear` 会删除所选表中所有记录，`--drop` 仅删除所选表。

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
cli = Client(table_name='default', bind='~/.isearch/isearch.sock')

# 提交图片（返回接受处理的数量，不是新增记录数）
image_paths = ['./img1.jpg', './img2.png', './folder/']
n_added = cli.add_images(image_paths, update_label=False, dedup_mode='fast')
print(f'已提交 {n_added} 张图片进行处理')

# 以图搜图（返回包含 id、label、paths、score 的记录列表或 None）
results = cli.search('./query.jpg', num=5, similarity=80)
if results:
    for row in results or []:
        print(row["id"], row["label"], row["paths"], row["score"])
else:
    print('无匹配结果或搜索队列满')

# 以文字搜图
results = cli.search('red apple', num=10, similarity=0)
for row in results or []:
    print(row["id"], row["label"], row["paths"], row["score"])

# 比较相似度（返回 0-100 float）
similarity = cli.compare_images('./img1.jpg', './img2.jpg')
print(f'相似度: {similarity}%')

# 表操作
tables = cli.list_tables()
print(f'可用表: {tables}')

info = cli.get_table_info()
print(f'表信息: {info}')

# 清空表（返回 True/False）
cleared = cli.clear_table()
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
```

## LanceDB storage / 存储迁移

The work directory is one LanceDB database: `~/.isearch/default.lance`, `~/.isearch/albumd.lance`, etc. Use `isearch table` to manage named image tables and `-d` to select a table for add/search operations. `ImageModel` in `imgsearch.models` defines `id: str`, `label: str`, `paths: list[str]` (default `[]`), and `vector: Vector(dim)` (float32). IDs are xxHash64 hex digests of the stored little-endian float32 vector bytes. `Clip.embed_images()`, `embed_image()`, and `embed_text()` return NumPy vectors; search RPC returns plain `id/label/paths/score` dictionaries without vectors.

导入默认先按 label 跳过（`--dedup-mode fast`），strict 模式不按 label 跳过。实际新增时按 ID 合并 paths，不检查旧路径、不从其他 ID 转移路径。默认保留原 label，`--update-label` 才会更新它。向量按字节判同，不是文件哈希或相似图去重；设备、模型和推理批次造成的浮点差异可能产生不同 ID。同一表请继续使用相同模型。

LanceDB commits writes immediately. Information reports each table's row count and disk bytes, including retained versions. There is no fixed capacity or `--rebuild`. Use one writing service per root. Prune runs synchronously and blocks storage operations while repairing the selected table; it does not re-embed valid paths belonging to only one ID.

旧 LanceDB 的 `name/images.lance` 和旧 HNSW 目录不会被自动转换。先停止服务，再显式迁移到根库下未存在的表。迁移保留原始向量数值，按哈希合并记录，不重新推理；原整数 ID 和 next_id 不保留。同 ID 按旧整数 ID 顺序保留第一个 label，合并全部路径。旧绝对路径 label 拆成文件名和路径，旧名称 label 原样保留且 paths 为空。迁移通过临时表验证后发布，失败不发布，源目录始终保留。

```shell
# Optional: only needed to read trusted HNSW backups (mapping uses pickle).
pip install 'imgsearch[migration]'

isearch service stop
# Legacy LanceDB: vector dimension is inferred.
python -m imgsearch.migrate ~/.isearch/default ~/.isearch --table default
# HNSW: specify its original vector dimension.
python -m imgsearch.migrate /backup/hnsw ~/.isearch --table recovered --dim 512
isearch service start
```

Check counts and searches before removing old backups; merged duplicate vectors reduce row counts. Migrated IDs are hashes of the preserved vectors, so re-embedding may produce a different ID. New writes are not copied back to the old directories. Existing Python/RPC callers must update for the new search result shape, `add_images(..., update_label=...)`, and hash IDs. `-l path/name` has been removed.

For this development environment, preserve the locally compiled packages already installed from `~/src/package/simple`:

```shell
uv sync --all-extras --no-install-package torch --no-install-package torchvision --no-install-package lancedb --inexact
```
