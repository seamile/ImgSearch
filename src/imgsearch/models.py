"""Image records and their canonical vector representation."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Self, TypedDict, cast

import numpy as np
import numpy.typing as npt
import xxhash
from lancedb.pydantic import FixedSizeListMixin, LanceModel, Vector
from pydantic import Field, create_model, field_validator, model_validator

Feature = npt.NDArray[np.float32]


def as_feature(value: npt.ArrayLike, dim: int) -> Feature:
    """Validate a finite nonzero vector and return contiguous little-endian float32 data."""
    vector = np.ascontiguousarray(np.asarray(value, dtype='<f4'))
    if vector.shape != (dim,) or not np.isfinite(vector).all():
        raise ValueError(f'Feature must contain {dim} finite numbers')
    if not np.any(vector):
        raise ValueError('Cosine features must be nonzero')
    return vector


class ImageInfo(TypedDict):
    id: str
    label: str
    paths: list[str]


class SearchResult(ImageInfo):
    score: float


def search_result(row: dict) -> SearchResult:
    """Convert a table search row into image metadata with a similarity score."""
    return SearchResult(
        id=row['id'],
        label=row['label'],
        paths=row['paths'],
        score=round((1.0 - float(row['_distance'])) * 100, 1),
    )


class ImageModel(LanceModel):
    id: str = Field(pattern=r'^[0-9a-f]{16}$')
    label: str = Field(min_length=1)
    paths: list[str] = Field(default_factory=list)
    vector: Vector(512, nullable=False)  # type: ignore

    @classmethod
    def dimension(cls) -> int:
        """Return the vector dimension declared by this model."""
        return cast(type[FixedSizeListMixin], cls.model_fields['vector'].annotation).dim()

    @field_validator('paths')
    @classmethod
    def normalize_paths(cls, paths: list[str]) -> list[str]:
        """Expand and deduplicate absolute paths while preserving symlink filenames."""
        if any(not path for path in paths):
            raise ValueError('Paths must be nonempty strings')
        # Keep symlink filenames while collapsing relative paths and dot segments.
        return list(dict.fromkeys(os.path.abspath(os.path.expanduser(path)) for path in paths))

    @model_validator(mode='after')
    def validate_identity(self) -> Self:
        """Validate the vector and verify that its hash matches the image ID."""
        vector = as_feature(self.vector, self.dimension())
        if self.id != xxhash.xxh64_hexdigest(vector.tobytes()):
            raise ValueError('Image ID does not match its vector')
        return self

    @classmethod
    def from_feature(cls, feature: npt.ArrayLike, *, label: str, paths=()) -> Self:
        """Create an image record with an ID computed from the canonical vector bytes."""
        vector = as_feature(feature, cls.dimension())
        return cls(id=xxhash.xxh64_hexdigest(vector.tobytes()), label=label, paths=list(paths), vector=vector)

    @classmethod
    def from_path(cls, feature: npt.ArrayLike, filepath: str | Path) -> Self:
        """Create an image record using the file stem as its label and the file path as its location."""
        return cls.from_feature(feature, label=Path(filepath).stem, paths=[str(filepath)])

    def merge(self, incoming: Self, *, update_label: bool = False) -> Self:
        """Merge paths for identical vectors and optionally replace the existing label."""
        if self.id != incoming.id or self.vector_bytes() != incoming.vector_bytes():
            raise ValueError('Conflicting vectors for image ID')
        return self.model_copy(
            update={
                'label': incoming.label if update_label else self.label,
                'paths': list(dict.fromkeys([*self.paths, *incoming.paths])),
            }
        )

    def vector_bytes(self) -> bytes:
        """Return the canonical vector bytes used to compute the image ID."""
        return as_feature(self.vector, self.dimension()).tobytes()


@lru_cache
def image_model(dim: int = 512) -> type[ImageModel]:
    """Return a cached image model with the requested vector dimension."""
    if dim <= 0:
        raise ValueError('dim must be positive')
    if dim == 512:
        return ImageModel
    return create_model(f'ImageModel{dim}', __base__=ImageModel, vector=(Vector(dim, nullable=False), ...))
