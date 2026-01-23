import numpy as np
import yt
from scipy.spatial import KDTree
from yt.units import Msun, kiloparsec

import packingcubes.bounding_box as bbox
import packingcubes.data_objects as data_objects
import packingcubes.octree as octree
import packingcubes.packed_tree as optree
from packingcubes.configuration import get_test_data_dir_path

data_path = get_test_data_dir_path()
simname = "IllustrisTNG"
ill_path = data_path / simname
snapfile = ill_path / "snapshot_090.hdf5"

rng = np.random.default_rng(0xBA55ADE89)

centers = []
radii = []


def load_data(decimation_factor=10):
    ds = data_objects.GadgetishHDF5Dataset(name=simname, filepath=snapfile)
    ds._positions = ds._positions[: int(len(ds) / decimation_factor), :]
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
    # sph_nodes_entire,sph_nodes_partial = tree._get_nodes_in_sphere(
    #     center=center, radius=radius
    # )
    # sph_inds


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
