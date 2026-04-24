"""
Functions for creating and loading Cubes

Also contains a CLI to create and save a ParticleCubes/MultiCubes object from
a snapshot file

Functions
---------
    Cubes(dataset=dataset, particle_types=[..])
        Load if present or create ParticleCubes/MultiCubes object from the
        provided dataset with the provided particle_types
    make_cubes(dataset=dataset, cubes_per_side=-1, save_dataset=False)
        Make a cubes_dict type structure from the provided dataset without saving
    make_ParticleCubes(dataset=dataset)
        Explicitly create a ParticleCubes object, raising an error if multiple
        particle types are present. Primarily intended for typechecking
    load_cubes(dataset=dataset)
        Load a cubes_dict type structure from the provideed dataset

"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Collection
from typing import cast

import h5py  # type: ignore
import numpy as np
from numba import get_num_threads  # type: ignore
from numba.typed import List
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
from packingcubes.bounding_box import BoundingBox, make_bounding_box
from packingcubes.cubes.cubes_numba import _prune_empty, cube, make_trees
from packingcubes.cubes.multi_cubes import MultiCubes
from packingcubes.cubes.particle_cubes import (
    ParticleCubes,
    has_cubes,
)
from packingcubes.data_objects import (
    DataContainer,
    Dataset,
    GadgetishHDF5Dataset,
    HDF5Dataset,
    InMemory,
    MultiParticleDataset,
)
from packingcubes.octree import _DEFAULT_PARTICLE_THRESHOLD
from packingcubes.packed_tree import (
    PackedTree,
)

LOGGER = logging.getLogger(__name__)


# from https://stackoverflow.com/a/27434050
class _LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()

        # parse arguments in the file and store them in a blank namespace
        data = parser.parse_args(contents.split(), namespace=None)
        for k, v in vars(data).items():
            # set arguments in the target namespace if they have not been set yet
            if getattr(namespace, k, None) is None:
                setattr(namespace, k, v)


def _process_args(argv=None):
    if argv is None:
        # need to skip caller or it's picked up as the snapshot file
        argv = sys.argv[1:]

    description = """
    Run the packingcubes program on a snapshot file.
    
    Default is to use the bounding box provided by the simulation,
    so if that's sufficient you do not need to provide x/y/z/dx/dy/dz
    """
    epilog = """
    If particle types are specified (using -t or --particle-types), the 
    snapshot file should be specified with -- SNAPSHOT OUTPUT at the end.
    
    Additional arguments can be read from a file by specifying the file with
    `@filename` anywhere among the argument string. Any arguments found in the
    file will overwrite previously specified arguments and be overwritten by 
    arguments specified later. Note that this is different behavior from the 
    -c/--config argument!
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase output verbosity"
    )
    parser.add_argument(
        "-n",
        "--side-length",
        default=-1,
        help="""
        number of cells per side [3-32], default -1 means use the lowest number
        of cells such that n**3 > # of available threads
        """,
        type=int,
        dest="n",
    )
    parser.add_argument(
        "-p",
        "--particle-threshold",
        help=(
            "the maximum number of particles per octree leaf."
            f"\nDefault: {_DEFAULT_PARTICLE_THRESHOLD}"
        ),
        default=_DEFAULT_PARTICLE_THRESHOLD,
        type=int,
    )
    box_args = parser.add_argument_group(
        "Box parameters", "Arguments to override parts of the default bounding box"
    )
    for ax in "xyz":
        box_args.add_argument(
            f"-{ax}", help=f"minimum bounding box {ax} coordinate", type=float
        )
        box_args.add_argument(
            f"-d{ax}", help=f"bounding box size in {ax} direction", type=float
        )
    conf_arg = parser.add_argument(
        "-c",
        "--config",
        help=(
            "Read in specified config file for arguments (CLI arguments will override)"
        ),
        type=open,
        action=_LoadFromFile,
    )
    parser.add_argument(
        "-t",
        "--particle-types",
        help=(
            "Names of particles to include "
            "(Can be integers or strings, 0 <=> PartType0)"
        ),
        nargs="+",
    )
    parser.add_argument("snapshot", help="Path to the snapshot file", type=str)
    parser.add_argument(
        "output",
        help="""
        Name of hdf5 file to save cubes information to. If not specified, cubes
        information will be discarded!
        """,
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--force-overwrite",
        help="Flag to overwrite cubes data contained in OUTPUT",
        action="store_true",
    )
    parser.add_argument(
        "--no-saving-dataset",
        help="""
        Don't save sorted particle positions and shuffle lists
        Normally sorted particle positions/shuffle lists are saved within a
        sidecar file to the snapshot. This flag disables that behavior.
        """,
        action="store_true",
    )

    args = parser.parse_args(argv)
    if args.n != -1 and (args.n < 3 or 32 < args.n):
        raise ValueError(
            f"Cubes per side needs to be within [3, 32]. You provided {args.n}"
        )
    if args.particle_types is not None:
        for i, n in enumerate(args.particle_types):
            try:
                args.particle_types[i] = f"PartType{int(n)}"
            except ValueError:
                continue
    if args.verbose >= 2:
        loglvl = logging.DEBUG
    elif args.verbose >= 1:
        loglvl = logging.INFO
    else:
        loglvl = LOGGER.level
    LOGGER.setLevel(loglvl)
    return args


def _process_box(*, dataset: Dataset, args) -> BoundingBox:
    box = dataset.bounding_box.box

    for i, ax in enumerate("xyz"):
        x: float | None = getattr(args, ax)
        if x is not None:
            box[i] = x
        dx: float | None = getattr(args, f"d{ax}")
        if dx is not None:
            box[3 + i] = dx

    return make_bounding_box(box)


def _get_cube_boxes(
    *, data: DataContainer, box: BoundingBox, cubes_per_side: int
) -> list[BoundingBox]:
    cube_size = box.size / cubes_per_side
    cube_boxes = List.empty_list(bbox.bbn_type)
    for i in range(cubes_per_side):
        for j in range(cubes_per_side):
            for k in range(cubes_per_side):
                cube_pos = box.position + [i, j, k] * cube_size
                cube = bbox.BoundingBox(np.hstack((cube_pos, cube_size)))
                cube_boxes.append(cube)
    cube_boxes.append(data.bounding_box)  # don't forget leftovers cube!
    return cube_boxes


def _process_cubes_per_side(cubes_per_side: int):
    if cubes_per_side > 0:
        return cubes_per_side
    nthreads = get_num_threads()
    cubes_per_side = 3
    flag = cubes_per_side**3 + 1 < nthreads
    while cubes_per_side < 32 and flag:
        cubes_per_side += 1
        flag = cubes_per_side**3 + 1 < nthreads
    return cubes_per_side


def make_cubes(
    *,
    dataset: MultiParticleDataset,
    cubes_per_side: int = -1,
    cube_box: bbox.BoxLike | None = None,
    particle_threshold: int | None = None,
    particle_types: str | Collection[str] | None = None,
    save_dataset: bool = False,
    **kwargs,
) -> dict[str, dict[str, NDArray | list[bbox.BoundingBox] | list[PackedTree]]]:
    """
    Create a cubes_dict from the provided dataset

    Parameters
    ----------
    dataset: MultiParticleDataset
        The dataset containing particle data. Will be sorted in-place, but will
        not save updated positional information unless save_dataset is True

    cubes_per_side: int, optional
        Number of cubes on a side. Dataset will be divided into `cubes_per_side`**3
        cubes, plus an additional cube to catch any remaining particles (if the
        cube_box is smaller than the actual data extants). Note: due to the
        `PackedTree`'s packed format, cubes must contain fewer than ~4 billion
        particles. If `cubes_per_side` is too small to support this, a `ValueError`
        will be raised. The limit is per-particle-type.

    cube_box: BoxLike, optional
        A box-like object (i.e. something that can convert to a `(6,)` ndarray)
        that delineates the region of data to be cubed. Any particles outside
        this region will fall into an overflow cube. Useful for zoom-in
        simulations or other datasets with sparse outer regions. Default is the
        data bounding box

    particle_threshold: int, optional
        Maximum number of particles in a tree leaf node. Default is `400`

    particle_types: Collection[str], optional
        Collection of particle types to include. Default is `dataset.particle_types`

    save_dataset: bool, optional
        Whether to save the sorted dataset positions out to a file. The data
        will be sorted in memory either way. Default `False`.

    Returns
    -------
    cubes_dict: dict
        A dictionary with 3 components:

         1. cube_indices - contains the data offsets for each cube's
            particles (i.e. cube 0 is from `cubes_indices[0]:cubes_indices[1]`)
         2. cube_boxes - containes the bounding box for each cube
         3. cube_trees - contains the `PackedTree` for each cube

    Raises
    ------
    ValueError
        If requested particle types aren't in the dataset or if too few cubes
        were requested for the number of particles

    """
    cubes = {}

    particle_types = (
        [particle_types] if isinstance(particle_types, str) else particle_types
    )
    if particle_types is None:
        requested_types = set(dataset.particle_types)
        particle_numbers = np.array(list(dataset.particle_numbers.values()))
        LOGGER.info(f"Found particle types: {dataset.particle_types}")
    else:
        requested_types = set(particle_types)
        data_types = set(dataset.particle_types)
        if not requested_types <= data_types:
            raise ValueError(
                f"Requested {requested_types - data_types} "
                "but it is not present in the dataset "
                f"({data_types})."
            )
        LOGGER.info(f"Using {requested_types}. Skipping {data_types - requested_types}")
        particle_numbers = np.array(
            [dataset.particle_numbers[i] for i in requested_types]
        )

    cube_box = dataset.bounding_box if cube_box is None else cube_box
    cube_box = bbox.make_bounding_box(cube_box)

    cubes_per_side = _process_cubes_per_side(cubes_per_side)

    num_cubes = cubes_per_side**3 + 1
    if np.any(particle_numbers / num_cubes > 2**31):
        raise ValueError(
            f"Insufficient number of cubes specified. Requested {cubes_per_side}"
            f" cubes per side, leading to >={particle_numbers / num_cubes}"
            f" particles per cube. Max per cube supported is {2**32}"
        )

    particle_threshold = (
        _DEFAULT_PARTICLE_THRESHOLD
        if particle_threshold is None
        else particle_threshold
    )

    for pt in requested_types:
        LOGGER.info(f"Processing {pt}")
        dataset.particle_type = pt
        data = dataset.data_container

        LOGGER.info("Cubing")
        cube_indices = cube(data, cubes_per_side, cube_box)

        # check cube sizes
        if np.any(np.diff(cube_indices[:-1]) >= 2**32):
            raise ValueError(
                f"Requested number of cubes is insufficient for {pt}. At least"
                " one cube has more than 2**32 particles, the max allowed for"
                " a packed tree."
            )

        LOGGER.info("Getting boxes")
        cube_boxes = _get_cube_boxes(
            data=data, box=cube_box, cubes_per_side=cubes_per_side
        )

        LOGGER.info("Removing empties")
        cube_indices, cube_boxes = _prune_empty(len(data), cube_indices, cube_boxes)

        LOGGER.info("Making trees")
        trees = make_trees(data, cube_indices, cube_boxes, particle_threshold)

        LOGGER.info("Converting to PackedTrees")
        ptrees = [
            PackedTree(
                source=t,
                particle_threshold=particle_threshold,
                bounding_box=b,
            )
            for t, b in zip(trees, cube_boxes, strict=True)
        ]

        cubes[pt] = {
            "cube_indices": cube_indices,
            "cube_boxes": cube_boxes,
            "cube_trees": ptrees,
        }
        LOGGER.info(f"Done with {pt}")

        if save_dataset:
            dataset.save()

    return cubes


def load_cubes(
    dataset: str | MultiParticleDataset,
) -> dict[str, dict]:
    """Load cubes data from a dataset. See make_cubes for a description of the output"""
    cubes_dict = {}
    if isinstance(dataset, Dataset) and not isinstance(dataset, HDF5Dataset):
        raise NotImplementedError("We can only load Cubes from HDF5 datasets")
    if not has_cubes(dataset):
        raise ValueError("No cubes in provided dataset")
    filepath = dataset.filepath if isinstance(dataset, HDF5Dataset) else dataset
    with h5py.File(
        filepath,
    ) as file:
        cubes_group = file["cubes"]
        pts = list(cubes_group.keys())
        for pt in pts:
            cubes = cubes_group[pt]
            cube_indices = cubes["indices"]
            number = cubes["number"]
            cube_boxes = []
            cube_trees = []
            for i in range(number):
                cube_boxes.append(bbox.make_bounding_box(cubes[f"box_{i}"]))
                cube_trees.append(PackedTree(source=cubes[f"tree_{i}"]))
            cubes_dict[pt] = {
                "cube_indices": cube_indices,
                "cube_boxes": cube_boxes,
                "cube_trees": cube_trees,
            }
    return cubes_dict


class CubesError(Exception):
    """Error during cubes creation or traversal"""

    pass


def _has_trees(cubes_dict: dict[str, dict]) -> bool:
    """Check if cubes_dict has trees in it"""
    return all("cube_trees" in cubes for _, cubes in cubes_dict.items())


def _add_trees_to_cubes_dict(
    *, cubes_dict: dict[str, dict], dataset: MultiParticleDataset, **kwargs
):
    """Generate missing PackedTrees from dataset on per-particle-type basis"""
    particle_threshold = getattr(
        kwargs, "particle_threshold", _DEFAULT_PARTICLE_THRESHOLD
    )
    for pt, cubes in cubes_dict.items():
        dataset.particle_type = pt
        cube_indices = cast(NDArray, cubes["cube_indices"])
        cube_boxes = cast(list[bbox.BoundingBox], cubes["cube_boxes"])
        cube_trees = make_trees(
            data=dataset.data_container,
            cube_indices=cube_indices,
            cube_boxes=cube_boxes,
            particle_threshold=particle_threshold,
        )


def Cubes(
    *,
    dataset: str | NDArray | MultiParticleDataset | None = None,
    cubes_dict: dict[str, dict] | None = None,
    **kwargs,
) -> ParticleCubes | dict[str, ParticleCubes]:
    """Create Cubes objects

    Create a Cubes-type object from the provided data. For datasets with
    multiple particle types, return a dictionary with mappings between
    the particle types and the corresponding ParticleCubes object.

    As an alternative to a dataset, you can provide a dictionary containing
    cube data offsets, bounding boxes, and optionally PackedTrees as
    `cube_indices`, `cube_boxes`, and `cube_trees`.

    Parameters
    ----------
    dataset: str | NDArray | MultiParticleDataset, optional
        Dataset containing positional data. Will be used to create a new
        ParticleCubes (if array or Dataset with only one particle type) or
        MultiCubes (otherwise) object, including sorting. Must provide either
        this or cubes_dict, below.

    cubes_dict: dict[str, dict], optional
        Dictionary of particle type -> dictionary elements. Each
        sub-dictionary has 3 components:

         1. cube_indices - contains the data offsets for each cube's particles
           (i.e. cube 0 is from `cubes_indices[0]:cubes_indices[1]`)
         2. cube_boxes - containes the `BoundingBox` for each cube
         3. cube_trees - contains the `PackedTree` for each cube

    **kwargs
        Extra arguments to `make_cubes` and `ParticleCubes`. See
        [make_cubes][make_cubes] and [ParticleCubes][ParticleCubes] for a
        description.

    Returns
    -------
    ParticleCubes | dict[str, ParticleCubes]
        Cubes object or dictionary, with the type dependent on the number
        of particle types present.

    See Also
    --------
    [ParticleCubes][ParticleCubes], [MultiCubes][MultiCubes]
    """
    if cubes_dict is None and dataset is None:
        raise CubesError("Must provide either a cubes_dict or dataset!")
    dataset = (
        InMemory(positions=dataset) if isinstance(dataset, np.ndarray) else dataset
    )
    # we only want to load the dataset if we need to
    if cubes_dict is None:
        assert dataset is not None
        try:
            cubes_dict = load_cubes(dataset)
        except (NotImplementedError, ValueError):
            dataset = (
                GadgetishHDF5Dataset(filepath=dataset)
                if isinstance(dataset, str)
                else dataset
            )
            cubes_dict = make_cubes(dataset=dataset, **kwargs)
    else:
        if not _has_trees(cubes_dict):
            if dataset is None:
                raise CubesError("cubes_dict has no trees and dataset not provided")
            dataset = (
                GadgetishHDF5Dataset(filepath=dataset)
                if isinstance(dataset, str)
                else dataset
            )
            _add_trees_to_cubes_dict(cubes_dict=cubes_dict, dataset=dataset, **kwargs)
    if len(cubes_dict) == 1:
        cubes = next(iter(cubes_dict.values()))
        return ParticleCubes(**cubes, **kwargs)
    return MultiCubes(cubes_dict=cubes_dict, **kwargs)._cubes_dict


def make_ParticleCubes(**kwargs) -> ParticleCubes:
    """
    Make ParticleCubes object from dataset or raise an error

    Parameters
    ----------
    **kwargs
        Refer to [Cubes][Cubes] documentation for a list of all posssible
        arguments
    """
    cubes = Cubes(**kwargs)
    if not isinstance(cubes, ParticleCubes):
        raise CubesError(
            f"""
            Multiple particle types present. Please specify one of 
            {list(cubes.keys())} as particle_types="PARTICLE_TYPE".
            """
        )
    return cubes


def make_MultiCubes(**kwargs) -> MultiCubes:
    """
    Make MultiCubes object from dataset even if there is only one particle type

    Parameters
    ----------
    **kwargs
        Refer to [Cubes][Cubes] documentation for a list of all posssible
        arguments
    """
    cubes = Cubes(**kwargs)
    multi = MultiCubes(cubes_dict={})
    pt = "particles"
    if "dataset" in kwargs and isinstance(kwargs["dataset"], MultiParticleDataset):
        pt = kwargs["dataset"].particle_type
    multi._cubes_dict = {pt: cubes} if isinstance(cubes, ParticleCubes) else cubes
    return multi


def cli():
    """Run the CLI for cubes generation"""
    logging.basicConfig()
    args = _process_args()
    if args.output and has_cubes(args.output) and not args.force_overwrite:
        sys.exit(
            "Provided output file already contains cubes data and"
            " you did not specify --force-overwrite"
        )
    dataset = GadgetishHDF5Dataset(filepath=args.snapshot, sorted_filepath=args.output)
    box = _process_box(dataset=dataset, args=args)
    cubes_dict = make_cubes(
        dataset=dataset,
        cubes_per_side=args.n,
        cube_box=box,
        particle_threshold=args.particle_threshold,
        particle_types=args.particle_types,
        save_dataset=not args.no_save_dataset,
    )
    LOGGER.info(cubes_dict.keys())
    cubes = make_MultiCubes(dataset=dataset, cubes_dict=cubes_dict)
    if args.output:
        cubes.save(args.output)
    return


if __name__ == "__main__":
    sys.exit(cli())
