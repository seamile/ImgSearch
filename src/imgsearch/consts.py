from pathlib import Path

# The default model to use.
DEFAULT_MODEL = 'wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'
BATCH_SIZE = 50

# The paths for the database files.
BASE_DIR = Path.home() / '.isearch'
DB_NAME = 'default'
IDX_NAME = 'index.db'
MAP_NAME = 'mapping.db'
CAPACITY = 10000

# The name for the service and the socket file.
SERVICE_NAME = 'isearch.service'
UNIX_SOCKET = BASE_DIR / 'isearch.sock'
