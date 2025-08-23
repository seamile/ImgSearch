# iFinder

iFinder 是一款图片查找工具，可以用来以图搜图，也可以通过文字描述搜索图片。

您可以直接在命令行中使用，也可以把它融入到其他系统中作为一个图片搜索模块。

本项目使用 TinyCLIP 模型提取图片特征，数据存储和搜索使用 Hnswlib。

## 使用方法

### 1. 构建索引

初次使用时 iFinder 时需先构建图片索引，将您系统中的图片导入到索引中。

```shell
ifinder -a IMG_DIRS_or_FILES ...
```

### 2. 搜索图片


- 以图搜图:

    ```shell
    ifinder <image_path>
    ```

- 以关键词搜图:

    ```shell
    ifinder -k <keywords>
    ```

其他可选项：
- `-d <db_path>` 指定数据库文件的路径
- `-n <num>` 指定搜索结果数量
- `-t <num>` 指定线程数
- `-m <model>` CLIP 模型名称
