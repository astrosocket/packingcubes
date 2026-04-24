"""Metadata for PackedTrees

Contains the TreeMeta class definition along with several functions for
converting between TreeMetas and packed arrays. See the Packed Format Design
Specification for more details.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Buffer
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from xxhash import xxh3_64_intdigest

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


class TreeMeta(NamedTuple):
    """The metadata for a PackedTree"""

    creation_timestamp: np.float64
    """ Creation time as a UTC timestamp"""
    checksum: int
    """ Actual checksum """
    bounding_box: bbox.BoundingBox
    """ Bounding box used when creating the tree """
    particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD
    """ Particle threshold used when creating the tree """
    meta_size: int = 144
    """ Header size in bytes """
    field_type: np.dtype = np.dtype(np.uint32)
    """ Field type (e.g. uint32)"""
    packed_spec_version: str = "1.0.0"
    """ Current version of the packed metadata standard in semver format"""
    checksum_method: str = "xxh3_64_intdigest"
    """ Method for computing the checksum """


class TreeMetaError(Exception):
    """Tree metadata parsing and formatting errors"""

    pass


def create_metadata(
    box: bbox.BoundingBox,
    packed: NDArray[np.uint32],
    particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD,
) -> TreeMeta:
    """Create a tree metadata object from the provided info

    Parameters
    ----------
    box: BoundingBox
        The bounding box used to create the tree

    packed: NDArray[uint32]
        The packed tree data

    particle_threshold: int, optional
        The particle threshold to split leaves. Defaults to
        `optree._DEFAULT_PARTICLE_THRESHOLD`

    Returns
    -------
    :
        A TreeMeta object containing the tree's metadata
    """
    creation = np.float64(datetime.datetime.now(tz=datetime.UTC).timestamp())
    checksum = xxh3_64_intdigest(bytes(packed))
    return TreeMeta(
        field_type=packed.dtype,
        creation_timestamp=creation,
        checksum=checksum,
        bounding_box=box,
        particle_threshold=particle_threshold,
    )


def pack_metadata(
    metadata: TreeMeta, packed_tree: NDArray[np.uint32]
) -> NDArray[np.uint32]:
    """Pack tree metadata

    Parameters
    ----------
    metadata: TreeMeta
        The metadata of the tree

    packed_tree: NDArray
        The packed tree data

    Returns
    -------
    :
        Packed metadata
    """
    packed_meta: NDArray = np.zeros(
        (int(metadata.meta_size / np.dtype(metadata.field_type).itemsize),),
        dtype=metadata.field_type,
    )
    packed_meta[0] = metadata.meta_size
    version = np.array(
        [np.uint8(v) for v in metadata.packed_spec_version.split(".")], dtype=np.uint32
    )
    packed_meta[1] = np.uint32(
        (np.uint32(np.int8(4)) << 24)
        | (version[0] << 16)
        | (version[1] << 8)
        | (version[2])
    )

    def to_field(
        x: np.float32 | np.float64 | np.int64 | np.uint64 | NDArray,
    ) -> NDArray:
        return np.frombuffer(x.tobytes(), dtype=metadata.field_type.type)

    ft = metadata.field_type.type
    packed_meta[2:4] = to_field(
        np.float64(datetime.datetime.now(tz=datetime.UTC).timestamp())
    )
    packed_meta[4:21] = to_field(np.array(metadata.checksum_method))
    packed_meta[21:23] = to_field(np.uint64(xxh3_64_intdigest(packed_tree)))
    packed_meta[23:35] = to_field(metadata.bounding_box.box)
    packed_meta[35] = np.uint32(metadata.particle_threshold)
    return packed_meta


def extract_metadata(source: Buffer) -> tuple[TreeMeta, NDArray[np.uint32]]:
    """Extract the metadata and packed tree information from a buffer

    Parameters
    ----------
    source: Buffer
        A Buffer containing the packed data

    Returns
    -------
    metadata:
        The tree's metadata

    packed_tree:
        The actual packed tree data as a numpy array

    Raises
    ------
    ValueError
        If the metadata does not match an expected format

    NotImplementedError
        If an unimplemented data format is specified (currently only uint32)

    OctreeError
        If the metadata checksum does not match the computed checksum
    """
    combined = np.frombuffer(source, dtype=np.uint32)
    meta_size = combined[0]
    if meta_size != 144:
        # need to check endianness
        dts = np.dtype(np.uint32).newbyteorder("S")
        meta_size = np.asarray(meta_size, dtype=dts)
        if meta_size != 144:
            raise TreeMetaError(
                "Unknown metadata format! Expected first "
                f"field to be equal to 144, got {meta_size}"
            )
        combined = np.asarray(combined, dtype=dts)
    # extract field type and packed specification version
    ft_version = combined[1]
    ft = np.int8(ft_version >> 24)
    if ft != 4:
        raise NotImplementedError(
            f"Only uint32 (ft=4) is currently available (provided {ft})"
        )
    version_list = [0xFF & (ft_version >> i) for i in range(16, -1, -8)]
    version = f"{version_list[0]}.{version_list[1]}.{version_list[2]}"

    def from_field(x: NDArray, dt: np.dtype) -> NDArray:
        return np.frombuffer(x.tobytes(), dtype=dt)

    creation_time = np.float64(from_field(combined[2:4], np.dtype(np.float64)))
    checksum_method = str(from_field(combined[4:21], np.dtype("U17"))[0])
    if checksum_method != "xxh3_64_intdigest":
        LOGGER.warning(
            f"Unknown checksum method: {checksum_method}. Expected xxh3_64_intdigest"
        )
    checksum = int(from_field(combined[21:23], np.dtype(np.uint64)))
    box_array = from_field(combined[23:35], np.dtype(np.float64))
    box = bbox.make_bounding_box(box_array)
    particle_threshold = combined[35]
    packed_tree = combined[36:]
    computed_checksum = xxh3_64_intdigest(packed_tree)
    if checksum != computed_checksum:
        raise octree.OctreeError(
            f"Packed tree checksum {computed_checksum}"
            f" does not match header ({checksum})!"
        )
    metadata = TreeMeta(
        meta_size=meta_size,
        field_type=np.dtype(np.uint32),
        creation_timestamp=creation_time,
        checksum=checksum,
        bounding_box=box,
        particle_threshold=particle_threshold,
        checksum_method=checksum_method,
        packed_spec_version=version,
    )
    return metadata, packed_tree
