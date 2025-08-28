from pathlib import Path

# The default model to use.
DEFAULT_MODEL = 'wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M'
BATCH_SIZE = 10

# The paths for the database files.
BASE_DIR = Path.home() / '.pixa'
DB_NAME = 'default'
IDX_NAME = 'index.db'
MAP_NAME = 'mapping.db'
CAPACITY = 10000

# The name for the service and the socket file.
SERVICE_NAME = 'pixa.service'
UNIX_SOCKET = BASE_DIR / 'pixa.sock'
