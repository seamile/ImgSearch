from pathlib import Path
from pickle import dump, load  # noqa: S403

from hnswlib import Index

DEFAULT_INDEX = Path.home() / '.ifinder' / '.ifinder.db'


class ImgBase:
    """Image feature index"""

    def __init__(self, dim: int = 512, db_path: Path = DEFAULT_INDEX) -> None:
        self.dim = dim
        self.db_path = db_path
        self.idx_mapping: dict[int, str] = {}
        self.index: Index | None = None
        self.current_id = 0

        # Automatically load or create index file
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load index file, or create a new index if not exists"""
        try:
            if self.db_path.exists():
                # Load existing index
                with self.db_path.open('rb') as f:
                    data = load(f)  # noqa: S301

                self.idx_mapping = data['idx_mapping']
                self.current_id = data['current_id']

                # Restore index
                if data['index_data']:
                    self.index = Index(space='cosine', dim=data['dim'])
                    # Write to temp file and load
                    import tempfile

                    with tempfile.NamedTemporaryFile() as tmp:
                        Path(tmp.name).write_bytes(data['index_data'])
                        self.index.load_index(tmp.name)
                else:
                    self.new_index()
            else:
                # Create new index
                self.new_index()
        except Exception:
            # If loading fails, create new index
            self.new_index()

    def new_index(self):
        """Create a new index"""
        self.index = Index(space='cosine', dim=self.dim)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16)
        self.idx_mapping = {}
        self.current_id = 0

    def next_max_elements(self):
        """Get next max elements for resizing index"""
        if self.index is None:
            return 10000
        return round(self.index.max_elements * 1.6)

    def add_image(self, path: Path, feature: list[float]):
        """Add one image to index"""
        if self.index is None:
            self.new_index()

        # Check if we need to resize the index
        if self.current_id >= self.index.max_elements:
            self.index.resize_index(self.next_max_elements())

        # Add the feature vector to the index
        self.index.add_items([feature], [self.current_id])
        self.idx_mapping[self.current_id] = str(path)
        self.current_id += 1

    def add_images(self, paths: list[Path], features: list[list[float]]):
        """Add multiple images to index"""
        if not paths or not features or len(paths) != len(features):
            return

        if self.index is None:
            self.new_index()

        # Check if we need to resize the index
        needed_capacity = self.current_id + len(features)
        if needed_capacity > self.index.max_elements:
            new_size = max(self.next_max_elements(), needed_capacity)
            self.index.resize_index(new_size)

        # Prepare IDs and update mapping
        ids = list(range(self.current_id, self.current_id + len(features)))
        for i, path in enumerate(paths):
            self.idx_mapping[self.current_id + i] = str(path)

        # Add features to index
        self.index.add_items(features, ids)
        self.current_id += len(features)

    def search(self, feature: list[float], k: int = 10) -> list[tuple[str, float]]:
        """Search images by feature vector"""
        if self.index is None or self.current_id == 0:
            return []

        # Search for similar images
        labels, distances = self.index.knn_query([feature], k=min(k, self.current_id))

        # Convert results to (path, similarity) tuples
        results = []
        for label, distance in zip(labels[0], distances[0], strict=False):
            if label in self.idx_mapping:
                # Convert cosine distance to similarity (1 - distance)
                similarity = 1.0 - distance
                results.append((self.idx_mapping[label], similarity))

        return results

    @classmethod
    def load_db(cls, path: Path) -> 'ImgBase':
        """Load database from file"""
        if not path.exists():
            raise FileNotFoundError(f'Database file not found: {path}')

        with path.open('rb') as f:
            data = load(f)  # noqa: S301

        # Create new instance and restore data
        instance = cls(dim=data['dim'])
        instance.idx_mapping = data['idx_mapping']
        instance.current_id = data['current_id']

        # Restore index
        if data['index_data']:
            instance.index = Index(space='cosine', dim=data['dim'])
            # Write index data to temporary file and load
            import tempfile

            with tempfile.NamedTemporaryFile() as tmp:
                Path(tmp.name).write_bytes(data['index_data'])
                instance.index.load_index(tmp.name)

        return instance

    def save(self, path: Path | None = None):
        """Save database to file"""
        if path is None:
            path = self.db_path

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        data = {'dim': self.dim, 'idx_mapping': self.idx_mapping, 'current_id': self.current_id, 'index_data': None}

        # Save index data if exists
        if self.index is not None and self.current_id > 0:
            # Save index to temporary file and read as bytes
            import tempfile

            with tempfile.NamedTemporaryFile() as tmp:
                self.index.save_index(tmp.name)
                data['index_data'] = Path(tmp.name).read_bytes()

        # Save to file
        with path.open('wb') as f:
            dump(data, f)
