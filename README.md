# Pixa

Pixa 是一款轻量级的图片搜索引擎，可以用来以图搜图，也可以通过文字描述搜索图片。他的资源占用很低，可以部署在性能有限的机器上。您可以直接在命令行中使用，也可将它融入到其他系统中作为一个图片搜索模块使用。

本项目使用 TinyCLIP 模型提取图片特征，数据存储和搜索使用 HNSWlib。程序整体采用 C-S 架构，使用前需要先启动服务，然后通过客户端进行操作。

## 使用方法

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

将指定路径的图片添加到数据库中。当指定的目标是文件夹时，会添加其中的所有图片。

```shell
px -a ./foo/img1.png ./bar/img2.jpg ./path/to/images_dir/
```

### 3. 搜索图片

通过样本图片或描述信息搜索图片。可通过 `-n` 参数指定搜索结果数量，搜索结果会按相似度从高到低排序。

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
- `-m MODEL`    CLIP 模型名称
- `-n NUM`      搜索结果数量（默认: 10）

### 6. 作为模块导入

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
