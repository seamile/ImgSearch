from pathlib import Path

MODELS = {
    'ViT-8Y': ('TinyCLIP-ViT-8M-16-Text-3M', 'YFCC15M'),  # minimal model
    'RN-19L': ('TinyCLIP-ResNet-19M-Text-19M', 'LAION400M'),
    'ViT-22L': ('TinyCLIP-auto-ViT-22M-32-Text-10M', 'LAION400M'),
    'RN-30L': ('TinyCLIP-ResNet-30M-Text-29M', 'LAION400M'),
    'ViT-39Y': ('TinyCLIP-ViT-39M-16-Text-19M', 'YFCC15M'),
    'ViT-40L': ('TinyCLIP-ViT-40M-32-Text-19M', 'LAION400M'),
    'ViT-45L': ('TinyCLIP-auto-ViT-45M-32-Text-18M', 'LAION400M'),
    'ViT-45LY': ('TinyCLIP-auto-ViT-45M-32-Text-18M', 'LAIONYFCC400M'),  # most balanced
    'ViT-61L': ('TinyCLIP-ViT-61M-32-Text-29M', 'LAION400M'),
    'ViT-63L': ('TinyCLIP-auto-ViT-63M-32-Text-31M', 'LAION400M'),
    'ViT-63LY': ('TinyCLIP-auto-ViT-63M-32-Text-31M', 'LAIONYFCC400M'),  # best-performing
}
# The default model to use.
DEFAULT_MODEL_KEY = 'ViT-8Y'
BATCH_SIZE = 100

# The paths for the database files.
BASE_DIR = Path.home() / '.isearch'
DB_NAME = 'default'
IDX_NAME = 'index.db'
MAP_NAME = 'mapping.db'
CAPACITY = 10000

# The name for the service and the socket file.
SERVICE_NAME = 'isearch.service'
UNIX_SOCKET = BASE_DIR / 'isearch.sock'
