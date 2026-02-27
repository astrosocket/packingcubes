import contextlib
import pickle
from functools import partial

import numpy as np
import yt
from scipy.spatial import KDTree
from yt.units import Msun, kiloparsec

import packingcubes.bounding_box as bbox
import packingcubes.cubes as cubes
import packingcubes.data_objects as data_objects
import packingcubes.octree as octree
import packingcubes.packed_tree as optree
from packingcubes.configuration import get_test_data_dir_path

data_path = get_test_data_dir_path()
simname = "IllustrisTNG"
ill_path = data_path / simname
snapfile = ill_path / "snapshot_090.hdf5"
particle_type = "PartType0"

rng = np.random.default_rng(0xBA55ADE89)

centers = []
radii = []


def load_data(decimation_factor=10):
    ds = data_objects.GadgetishHDF5Dataset(name=simname, filepath=snapfile)
    # ds._positions = ds._positions[: int(len(ds) / decimation_factor), :]
    if ds.particle_type != particle_type:
        with contextlib.suppress(data_objects.DatasetError):
            ds.particle_type = particle_type
    ds._positions = ds._positions[:: int(decimation_factor), :]
    try:
        del ds._index
    finally:
        ds._setup_index()
    min_bounds = np.min(ds.positions, axis=0)
    max_bounds = np.max(ds.positions, axis=0)
    ds._box = bbox.make_bounding_box(np.hstack((min_bounds, max_bounds - min_bounds)))
    random_search_balls(ds)
    return ds


def reset_data(ds):
    original_inds = np.argsort(ds.index)
    ds._positions = ds._positions[original_inds, :]
    del ds._index
    ds._setup_index()
    return ds


def random_search_balls(ds):
    box = ds.bounding_box
    for _ in range(100):
        centers.append(rng.random(box.size.size) * box.size + box.position)
        radii.append(10 ** (rng.random() * np.log10(rng.choice(box.size))))


def python_octree_creation(ds):
    return octree.PythonOctree(
        dataset=ds,
    )


def python_octree_query_ball_point(tree: octree.Octree):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )


def packed_octree_creation(ds):
    return optree.PackedTree(dataset=ds)


def packed_octree_query_ball_point(tree: optree.PackedTree):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )


# we want the PackedTree stuff to be pre-compiled
def precompile():
    packed_octree_query_ball_point(packed_octree_creation(load_data(1e4)))


def kdtree_creation(ds):
    return KDTree(data=ds.positions, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD)


def kdtree_query_ball_point(tree: KDTree):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.query_ball_point(x=c, r=r)


def yt_setup(decimation_factor=10):
    ds = load_data(decimation_factor=decimation_factor)
    yt.set_log_level("warning")
    ppx, ppy, ppz = ds.positions[:, 0], ds.positions[:, 1], ds.positions[:, 2]
    data = {
        ("io", "particle_position_x"): ppx,
        ("io", "particle_position_y"): ppy,
        ("io", "particle_position_z"): ppz,
        ("io", "particle_mass"): np.ones_like(ds.index),
    }
    box = ds.bounding_box.box
    bbox = np.zeros((3, 2))
    bbox[:, 0] = box[:3]
    bbox[:, 1] = box[:3] + box[3:]
    return yt.load_particles(
        data, length_unit=1 * kiloparsec, mass_unit=1 * Msun, bbox=bbox
    )


def yt_creation(ytdata):
    return ytdata.sphere((centers[0], "kpc"), (radii[0], "kpc"))


def yt_search(sph):
    sph_inds = sph["io", "particle_mass"]


def remove_problem_classes_from_state(self):
    state = self.__dict__.copy()
    data_keys = [
        n
        for n, v in state.items()
        if isinstance(v, (data_objects.Dataset, data_objects.DataContainer))
    ]
    for k in data_keys:
        del state[k]
    fixed_jitclass_sizes = {
        "BoundingBox": 48 + 184,
    }
    jitclass_keys = {
        n: str(type(v))
        for n, v in state.items()
        if isinstance(v, (bbox.BoundingBox, optree.PackedTreeNumba))
    }
    for k, v in jitclass_keys.items():
        # v looks something like "<class 'numba.XXX.YYY.BoundingBox'>"
        # so using a regex would be better, but this is fine
        name = v.split(".")[-1][:-2]
        if name in fixed_jitclass_sizes:
            # bytes has a minimum of 4 elements = 4 bytes + 33 bytes overhead
            state[k] = bytes(np.maximum(fixed_jitclass_sizes[name] - 33, 0))
        elif name == "PackedTreeNumba":
            # PackedTreeNumba has 3 fields:
            #   particle_threshold - 4 byte int,
            #   tree - 128 bytes + 4 bytes*length
            #   data - not included
            tree_len = len(state[k].tree)
            state[k] = bytes(np.maximum(4 + (128 + 4 * tree_len) - 33, 0))

    return state


def tree_sizes(decimation_factor=10):
    # print(".Precompiling") # noqa
    # precompile()
    # print(".Loading data")  # noqa
    ds = load_data(decimation_factor=decimation_factor)
    b = pickle.dumps(ds.positions)
    print(f"Positions: {len(b)}")  # noqa

    tcf_dict = {
        "python": python_octree_creation,
        "packed": packed_octree_creation,
        "kdtree": kdtree_creation,
    }
    # print(".Creating trees and computing memory usage")  # noqa
    for name, tree_creation_func in tcf_dict.items():
        tree = tree_creation_func(ds)
        match name:
            case "python":
                for node in tree:
                    node.__getstate__ = partial(remove_problem_classes_from_state, node)
            case "packed":
                tree.__getstate__ = partial(remove_problem_classes_from_state, tree)
        b = pickle.dumps(tree)
        print(f"{name}: {len(b)}")  # noqa


def cubing_setup():
    dataset = load_data(1)
    args = cubes._process_args(["-t0", "--", str(dataset.filepath)])
    box = cubes._process_box(dataset=dataset, args=args)
    return (dataset, args, box)


def cubing(setup):
    dataset, args, box = setup
    return cubes.Cubes(
        dataset=dataset,
        cubes_per_side=args.n,
        cube_box=box,
        particle_threshold=args.particle_threshold,
        particle_types=args.particle_types,
    )


def cubes_query_ball_points(cubes):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = cubes.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )
