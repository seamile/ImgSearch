"""Configuration Constants for ImgSearch

This module defines all configuration constants used throughout the application,
including OpenCLIP-compatible TinyCLIP model variants, batch sizes, database paths, service names, and
performance parameters. Constants are designed for easy customization while
maintaining reasonable defaults for typical use cases.

Model Selection Guide:

The available TinyCLIP models vary in size, accuracy, and performance. Choose based on
your hardware, accuracy needs, and inference speed requirements. Official benchmarks
(IN-1K Acc@1 on ImageNet-1K, MACs for compute, Throughput in pairs/second):

| Model    | IN-1K Acc@1(%) | MACs(G) | Throughput(pairs/s) |
|----------|---------------:|--------:|--------------------:|
| ViT-8Y   |           41.1 |     2.0 |               4,150 |
| ViT-22L  |           53.7 |     1.9 |               5,504 |
| RN-19L   |           56.4 |     4.4 |               3,024 |
| RN-30L   |           59.1 |     6.9 |               1,811 |
| ViT-40L  |           59.8 |     3.5 |               4,641 |
| ViT-45L  |           61.4 |     3.7 |               3,682 |
| ViT-45LY |           62.7 |     1.9 |               3,685 |
| ViT-61L  |           62.4 |     5.3 |               3,191 |
| ViT-39Y  |           63.5 |     9.5 |               1,469 |
| ViT-63L  |           63.9 |     5.6 |               2,905 |
| ViT-63LY |           64.5 |     5.6 |               2,909 |

*Data from [TinyCLIP Model Zoo](https://github.com/wkcn/TinyCLIP?tab=readme-ov-file#model-zoo).*
*Note: Models loaded via OpenCLIP; verify compatibility in [OpenCLIP docs](https://github.com/mlfoundations/open_clip).*

- **Small/Fast Models** (ViT-8Y, RN-19L, ViT-22L): Lower accuracy (41-56%), but high throughput
  (>3k pairs/s) and low compute (<5G MACs). Ideal for low-resource devices, mobile, or real-time
  applications where speed > precision.

- **Balanced Models** (ViT-40L, ViT-45LY, ViT-61L): Good accuracy (59-63%), moderate speed
  (3-4k pairs/s), ~3-5G MACs. DEFAULT_MODEL_KEY='ViT-45LY' recommended for most users -
  optimal tradeoff for desktop/GPU setups.

- **High-Accuracy Models** (ViT-39Y, ViT-63L/Y): Best precision (63-65%), but slower
  (1.5-3k pairs/s) and higher compute (5-10G MACs). Use for production search with sufficient
  GPU resources where recall is critical.

Database Structure:
- Each database is a directory under BASE_DIR containing:
  - index.db: HNSW vector index (binary).
  - mapping.db: Label-ID bidirectional mapping (pickled dict).
- Default capacity 10k items; auto-resizes in chunks of CAPACITY.
"""

from pathlib import Path

# CLIP Model Variants (name: (model_id, pretrained_weights, dim))
# Models from TinyCLIP family: lightweight alternatives to OpenAI CLIP
# Selection balances speed, accuracy, and resource usage
MODELS = {
    'ViT-8Y': ('TinyCLIP-ViT-8M-16-Text-3M', 'YFCC15M', 512),  # ~8M params, fastest, basic accuracy
    'RN-19L': ('TinyCLIP-ResNet-19M-Text-19M', 'LAION400M', 1024),
    'RN-30L': ('TinyCLIP-ResNet-30M-Text-29M', 'LAION400M', 1024),
    'ViT-22L': ('TinyCLIP-auto-ViT-22M-32-Text-10M', 'LAION400M', 512),
    'ViT-39Y': ('TinyCLIP-ViT-39M-16-Text-19M', 'YFCC15M', 512),
    'ViT-40L': ('TinyCLIP-ViT-40M-32-Text-19M', 'LAION400M', 512),
    'ViT-45L': ('TinyCLIP-auto-ViT-45M-32-Text-18M', 'LAION400M', 512),
    'ViT-45LY': ('TinyCLIP-auto-ViT-45M-32-Text-18M', 'LAIONYFCC400M', 512),  # Default: balanced accuracy/speed
    'ViT-61L': ('TinyCLIP-ViT-61M-32-Text-29M', 'LAION400M', 512),
    'ViT-63L': ('TinyCLIP-auto-ViT-63M-32-Text-31M', 'LAION400M', 512),
    'ViT-63LY': ('TinyCLIP-auto-ViT-63M-32-Text-31M', 'LAIONYFCC400M', 512),  # Best accuracy, most resource-intensive
}

# Default model: ViT-45LY - optimal balance for most use cases (speed/accuracy/resources)
DEFAULT_MODEL_KEY = 'ViT-45LY'

# Batch processing parameters
BATCH_SIZE = 100  # Images per batch for processing

# Database configuration
BASE_DIR = Path.home() / '.isearch'  # User home directory for DBs (~/.isearch)
DB_NAME = 'default'  # Default database name
IDX_NAME = 'index.db'  # HNSW vector index file
MAP_NAME = 'mapping.db'  # Label-ID bidirectional mapping file (pickled)
CAPACITY = 10000  # Initial index capacity; auto-resizes in increments of this value

# Service and networking
SERVICE_NAME = 'isearch.service'  # Pyro5 object ID for service lookup
UNIX_SOCKET = str((BASE_DIR / 'isearch.sock').resolve())  # Default UDS path for local connections
