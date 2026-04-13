import argparse
import contextlib
import logging
import pickle
import sys
import time
import timeit
from functools import partial
from typing import TextIO

import numpy as np
from numba import njit
from numba.typed import List
from numpy.typing import NDArray
from scipy.spatial import KDTree
from unyt import nanosecond, second, unyt_array, unyt_quantity

import packingcubes
import packingcubes.bounding_box as bbox
import packingcubes.cubes as cubes
import packingcubes.data_objects as data_objects
import packingcubes.octree as octree
import packingcubes.packed_tree as optree
from packingcubes.configuration import get_test_data_dir_path

LOGGER = logging.getLogger(__name__)

data_path = get_test_data_dir_path()
simname = "IllustrisTNG"
ill_path = data_path / simname
snapfile = ill_path / "snapshot_090.hdf5"
particle_type = "PartType0"

rng = np.random.default_rng(0xBA55ADE89)
_DEFAULT_QUERY_SIZE = 100


def _create_loading_pattern(filepath: str, loading_factor: int | None = None):
    """
    Create chunky slice loading if loading factor requested

    A "chunky" slice is similar to striding the data (e.g. ::10) except each
    slice is some number of particles thick. So effectively, normal striding
    would be chunky slicing with a chunk size of 1.

    Args:
        filepath: str
        Location of datafile

        loading_factor: int, optional
        The loading factor. Default is none

    Results:
        slices: list of slices or None
        List of chunky slices or None if loading_factor is None or 1
    """

    if loading_factor is None or loading_factor == 1:
        return None

    LOGGER.debug("Temporary data loading to determine length")
    ds = data_objects.GadgetishHDF5Dataset(
        name="", filepath=filepath, data_slices=np.s_[0:1]
    )
    num_particles = ds._particle_numbers[ds._particle_type]
    # want "chunky" slices - i.e. instead of every 10th particle out of 1000
    # grab 10 10-particle slices evenly distributed. So 0:10, 100:110, etc
    # use closest ints to sqrt(N/loading_factor) and evenly distribute
    # remainder
    num_loaded = num_particles / loading_factor
    num_chunks = np.floor(np.sqrt(num_loaded)).astype(int)
    num_skip = np.floor((num_particles - num_loaded) / num_chunks).astype(int)
    num_extra = num_loaded - num_chunks**2
    LOGGER.debug(
        f"""
        Loading data as {num_chunks} chunks of {num_chunks} particles, skipping
        {num_skip} particles between.
        """
    )
    loading_pattern = []
    offset = 0
    for i in range(num_chunks):
        loading_pattern.append(np.s_[offset : offset + num_chunks + (i <= num_extra)])
        offset += num_chunks + (i <= num_extra) + num_skip
    return loading_pattern


def load_data(
    decimation_factor=10,
    *,
    name: str = simname,
    filepath: str = snapfile,
    use_constant_number: int | None = None,
    number_balls: int = 100,
    loading_factor: int | None = None,
):
    LOGGER.debug("Beginning data loading")
    loading_pattern = _create_loading_pattern(filepath, loading_factor)
    ds = data_objects.GadgetishHDF5Dataset(
        name=name,
        filepath=filepath,
        data_slices=loading_pattern,
    )
    # ds._positions = ds._positions[: int(len(ds) / decimation_factor), :]
    if ds.particle_type != particle_type:
        with contextlib.suppress(data_objects.DatasetError):
            ds.particle_type = particle_type
    # Bug fix: convert to InMemory version so that when testing cubing versions
    # we don't try to reload the dataset (because cubing does every particle
    # type available by setting the particle type, which reloads the data,
    # undoing the decimation here. This means we don't need to manually set the
    # index or bounding box either, which is a nice benefit
    LOGGER.debug("Converting to InMemory")
    dataset = data_objects.InMemory(
        positions=ds._positions[:: int(decimation_factor), :]
    )
    loading_factor = loading_factor if loading_factor else 1
    LOGGER.info(
        f"Loaded {filepath} with loading factor x decimation factor = "
        f"{loading_factor * decimation_factor} => {len(dataset):.3e} particles"
    )
    if use_constant_number:
        if use_constant_number < 10:
            LOGGER.warning(
                "Specifying search balls with fewer than 10 particles."
                " I hope you're sure!"
            )
        centers, radii, particle_numbers = random_search_balls_constant_number(
            dataset, number_balls=number_balls, num_particles=use_constant_number
        )
    else:
        centers, radii, particle_numbers = random_search_balls_constant_volume(
            dataset, number_balls=number_balls
        )

    return dataset, centers, radii, particle_numbers


def set_decimation(
    *, ds: data_objects.Dataset, decimation_factor: int
) -> data_objects.Dataset:
    LOGGER.info(f"Decimating to {len(ds) / decimation_factor:.3e} particles")
    return data_objects.InMemory(
        positions=ds.positions.copy()[::decimation_factor, :],
    )


def reset_data(ds: data_objects.Dataset) -> data_objects.Dataset:
    original_inds = np.argsort(ds.index)
    ds.reorder(original_inds)
    # verify
    assert np.all(ds.index[: len(ds) - 1] == ds.index[1:] - 1), (
        "Some indices were not reset!"
    )
    return ds


def random_search_balls_constant_volume(ds, *, number_balls: int = 100):
    box = ds.bounding_box
    centers = []
    radii = []
    particle_numbers = []
    for _ in range(number_balls):
        centers.append(rng.random(box.size.size) * box.size + box.position)
        radii.append(10 ** (rng.random() * np.log10(rng.choice(box.size))))
    return centers, radii, particle_numbers


def random_search_balls_constant_number(
    ds: data_objects.Dataset,
    *,
    num_particles: int = 1000,
    number_balls: int = 100,
    error_threshold: float = 0.1,
    iterations_threshold: int = 20,
):
    LOGGER.info(
        f"Generating constant particle number search balls of size {num_particles}"
    )
    # kdtree = optree.KDTree(data=ds.positions)
    kdtree = KDTree(data=ds.positions, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD)
    box = ds.bounding_box
    bad_balls = 0
    centers = []
    radii = []
    particle_numbers = []
    for i in range(number_balls):
        # bisection method:
        # keep center generation and use radius generation for initial radii
        # assume r = 0 => n_enclosed = 0
        # bisect radii until |n_enclosed - num_particles|/num_particles < 5%
        center = rng.random(box.size.size) * box.size + box.position

        if num_particles > len(ds):
            # need to return a radii bigger than box -> just use twice max dx
            centers.append(center)
            r = max(ds.bounding_box.size)
            radii.append(r)
            particle_numbers.append(len(ds))
            continue

        r = 10 ** (rng.random() * np.log10(rng.choice(box.size)))
        LOGGER.debug(f"Starting search for ball {i} with center {center} and r0={r}")
        n_enclosed = len(
            kdtree.query_ball_point(
                center,
                r,
                # strict=True,
            )
        )
        rlower = 0

        # ensure we're starting from a big enough radii
        # can update lower limit at same time
        while n_enclosed < num_particles:
            LOGGER.debug(f"{r=:.3g} too small, only {n_enclosed} particles. Doubling")
            rlower = r
            r *= 2
            n_enclosed = len(
                kdtree.query_ball_point(
                    center,
                    r,
                    # strict=True,
                )
            )
        rupper = r
        LOGGER.debug(
            f"Starting refined search with rlower={rlower:.4g} and rupper={rupper:.4g}"
            f" and {n_enclosed} particles"
        )
        num_iterations = 0
        error = abs(n_enclosed - num_particles) / num_particles
        while num_iterations < iterations_threshold and error >= error_threshold:
            r = (rupper + rlower) / 2
            LOGGER.debug(
                f"Have {n_enclosed} particles ({error=:.3g}). Checking {r=:.4g}"
            )
            n_enclosed = len(
                kdtree.query_ball_point(
                    center,
                    r,
                )
            )  # strict=True))
            error = abs(n_enclosed - num_particles) / num_particles
            if n_enclosed > num_particles:
                rupper = r
            else:
                rlower = r
            num_iterations += 1
        LOGGER.debug(
            f"Finished with r = {r}, {error=:.3g}, and "
            f"{n_enclosed} particles after {num_iterations} iterations"
        )
        if num_iterations >= iterations_threshold:
            LOGGER.info(f"Taking too long, skipping sphere {i}")
            bad_balls += 1
            continue
        centers.append(center)
        radii.append(r)
        particle_numbers.append(n_enclosed)
    if bad_balls:
        LOGGER.debug(f"Skipped {bad_balls} spheres")
    if bad_balls > number_balls / 10:
        LOGGER.warning(
            f"More than 10% of spheres ({bad_balls}) were skipped."
            " Timing stats may suffer"
        )
    reset_data(ds)  # need to undo changes to get accurate timing later
    return centers, radii, particle_numbers


def python_octree_creation(ds):
    return octree.PythonOctree(
        dataset=ds,
    )


def python_octree_query_ball_point(tree: octree.Octree, *, centers, radii, **kwargs):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )


def packed_octree_creation(ds):
    return optree.PackedTree(dataset=ds)


def packed_octree_query_ball_point(
    tree: optree.PackedTree, *, centers, radii, **kwargs
):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )


def packed_octree_query_ball_point_indices(
    data: data_objects.DataContainer,
    tree: optree.PackedTree,
    *,
    centers,
    radii,
    **kwargs,
):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_index_list_in_sphere(
            data=data,
            center=c,
            radius=r,
            strict=True,
        )


@njit
def _packed_octree_qbp_jitted(
    tree: optree.packed_tree_numba.PackedTreeNumba,
    spheres: List[bbox.BoundingSphere],
    num_reps: int,
):
    for i in range(len(spheres)):
        sph = spheres[i]
        for _ in range(num_reps):
            sph_inds = tree._get_particle_indices_in_shape(sph)


def packed_octree_qbp_jitted(tree: optree.PackedTree, *, centers, radii, **kwargs):
    spheres = List.empty_list(bbox.bs_type)
    for c, r in zip(centers, radii, strict=True):
        sph = bbox.make_bounding_sphere(r, center=c, unsafe=True)
        spheres.append(sph)
    num_reps = 100000
    _packed_octree_qbp_jitted(tree._tree, spheres, num_reps)


def packed_kdtree_creation(ds):
    return optree.KDTree(data=ds, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD)


def packed_kdtree_query_ball_point(
    tree: optree.KDTree, *, centers, radii, particle_numbers: list[int], **kwargs
):
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        # Ensure kdtree output matches scipy's regardless of defaults
        sph_inds = tree.query_ball_point(
            x=c, r=r, strict=True, return_lists=False, return_data_indices=True
        )
        if particle_numbers and len(sph_inds) != particle_numbers[i]:
            raise ValueError(
                f"""
                Particle number mismatch: expected {particle_numbers[i]} particles
                for ball {i} and got {len(sph_inds)}.
                """
            )


def packed_kdtree_query(
    tree: optree.KDTree, *, centers, k=_DEFAULT_QUERY_SIZE, **kwargs
):
    for c in centers:
        dd, ii = tree.query(c, k=k, return_data_indices=True, return_sorted=True)


def brute_force_creation(ds):
    return ds.positions


def brute_force_search(
    positions, *, centers, radii, particle_numbers: list[int], **kwargs
):
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        mask = np.sum((positions - c) ** 2, axis=1) <= r**2
        number = np.sum(mask)
        if particle_numbers and number != particle_numbers[i]:
            raise ValueError(
                f"""
                Particle number mismatch: expected {particle_numbers[i]} particles
                for ball {i} and got {number}.
                """
            )


def scipy_kdtree_creation(ds):
    return KDTree(data=ds.positions, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD)


def scipy_kdtree_query_ball_point(
    tree: KDTree,
    *,
    centers,
    radii,
    particle_numbers: list[int],
    **kwargs,
):
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        sph_inds = tree.query_ball_point(x=c, r=r)
        if particle_numbers and len(sph_inds) != particle_numbers[i]:
            raise ValueError(
                f"""
                Particle number mismatch: expected {particle_numbers[i]} particles
                for ball {i} and got {len(sph_inds)}.
                """
            )


def scipy_kdtree_query(tree: KDTree, *, centers, k=_DEFAULT_QUERY_SIZE, **kwargs):
    for c in centers:
        dd, ii = tree.query(c, k=k)


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
        "kdtree": scipy_kdtree_creation,
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


def cubing_setup(decimation_factor: int = 1, *, dataset: data_objects.Dataset = None):
    if dataset is None:
        dataset = load_data(decimation_factor)
    # We'll use _process_args to get as close to the default behavior as we can
    # InMemory datasets only are PartType0 by default
    args = cubes._process_args(
        ["-t0", "--no-saving-dataset", "--", str(dataset.filepath)]
    )
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
        save_dataset=False,
    )


def cubes_query_ball_points(cubes, *, centers, radii, **kwargs):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = cubes.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )


def get_creation_search_dicts():
    creation_dict = {
        "pyoct": {
            "fun": "python_octree_creation",
        },
        "packed": {
            "fun": "packed_octree_creation",
        },
        "cubes": {
            "fun": "cubing",
            "setup": cubing_setup,
        },
        "kdtree": {"fun": "packed_kdtree_creation"},
        "scipy": {"fun": "scipy_kdtree_creation"},
        "brute": {"fun": "brute_force_creation"},
    }
    search_dict = {
        "pyoct": {
            "fun": "python_octree_query_ball_point(search_obj)",
            "tree": "pyoct",
        },
        "packed": {
            "fun": "packed_octree_query_ball_point(search_obj)",
            "tree": "packed",
        },
        "packli": {
            "fun": (
                "packed_octree_query_ball_point_indices(data, search_obj)"
            ),  # needs dataset + tree
            "tree": "packed",
        },
        "packnumb": {
            "fun": "packed_octree_qbp_jitted(search_obj)",
            "tree": "packed",
            "scaling": 100000,
        },
        "cubes": {
            "fun": "cubes_query_ball_points(search_obj)",
            "tree": "cubes",
        },
        "kdtree": {
            "fun": "packed_kdtree_query_ball_point(search_obj)",
            "tree": "kdtree",
        },
        "kdq": {
            "fun": "packed_kdtree_query(search_obj)",
            "tree": "kdtree",
        },
        "brute": {
            "fun": "brute_force_search(search_obj)",
            "tree": "brute",
        },
        "scipy": {
            "fun": "scipy_kdtree_query_ball_point(search_obj)",
            "tree": "scipy",
        },
        "sciq": {
            "fun": "scipy_kdtree_query(search_obj)",
            "tree": "scipy",
        },
    }
    return creation_dict, search_dict


def _format_time(times: unyt_array | unyt_quantity) -> unyt_array:
    # day and hour are more for fun I suppose. If the timing is measured in
    # days, there are bigger problems...
    units = ["d", "hr", "min", "s", "ms", "us", "ns"]
    if isinstance(times, unyt_quantity):
        times = unyt_array([times])
    mintime = np.abs(min(times))

    for unit in units:
        if mintime.to(unit) > 1:
            return times.to(unit)
    return times.to(units[-1])


def _run_timer(
    timer: timeit.Timer, *, add_scaling: float = 1
) -> tuple[int, unyt_array]:
    try:
        # do additional run for any missing precompile
        timer.timeit(1)
        number, _ = timer.autorange()
        time_vec = timer.repeat(number=number)
        time_vec *= second
        time_vec /= number  # change to per-loop
        time_vec /= add_scaling  # include any additional scaling (e.g. per-sphere)
    except ValueError as ve:
        timer.print_exc()
        LOGGER.warning(ve)
        number = -1
        time_vec = [-1, -1] * second
    return number, time_vec


def _run_creation_perf_timer(
    creation_fun: str, dataset: data_objects.MultiParticleDataset, setup_data
) -> tuple[unyt_quantity, unyt_array]:
    """
    Time search object creation using perf_timer
    """
    creation_fun = globals()[creation_fun]
    # do initial run to catch any precompilation issues and to prime data loading
    creation_fun(setup_data)
    # reset data
    reset_data(dataset)

    def timeit(
        creation_fun,
        dataset: data_objects.MultiParticleDataset,
        setup_data,
        number: int,
    ) -> unyt_quantity:
        t0 = time.perf_counter_ns()
        delta = 0
        for _ in range(number):
            t1 = time.perf_counter_ns()
            reset_data(dataset)
            delta += time.perf_counter_ns() - t1
            creation_fun(setup_data)
        tfinal = time.perf_counter_ns()
        return (tfinal - t0 - delta) * nanosecond

    # do manual autorange
    scale = 1
    done = False
    try:
        while not done:
            for i in [1, 2, 5]:
                number = i * scale
                time_taken = timeit(
                    creation_fun=creation_fun,
                    dataset=dataset,
                    setup_data=setup_data,
                    number=number,
                )
                if time_taken.to("s") >= 0.2:
                    done = True
                    break
            scale *= 10
    except ValueError as ve:
        LOGGER.warning(ve)
        return -1, [-1, -1] * second

    time_vec = unyt_array(np.full((5,), -1.0), "s")
    try:
        for i in range(len(time_vec)):
            time_vec[i] = timeit(
                creation_fun=creation_fun,
                dataset=dataset,
                setup_data=setup_data,
                number=number,
            ).to("s")
    except ValueError as ve:
        LOGGER.warning(ve)
        return -1, [-1, -1] * second
    return number, time_vec


def get_search_obj(
    *,
    function: str,
    dataset: data_objects.Dataset,
    creation_dict: dict,
    results: dict = None,
    dry_run: bool = False,
):
    cd = creation_dict[function]
    setup = cd.get("setup", lambda dataset: dataset)
    scaling = cd.get("scaling", 1)
    setup_data = setup(dataset=dataset)
    globals()["dataset"] = dataset
    globals()["setup_data"] = setup_data

    if results is not None:
        LOGGER.debug(f"Timing {function} creation")
        statement = f"reset_data(dataset);{cd['fun']}(setup_data)"
        if not dry_run:
            number, time_vec = _run_creation_perf_timer(
                creation_fun=cd["fun"], dataset=dataset, setup_data=setup_data
            )
        else:
            number = -1
            time_vec = [-1, -1] * second
        time_vec = _format_time(time_vec)
        results[function] = (min(time_vec), time_vec)
        test_name = (
            (function + f" ({cubes.nthreads} threads)")
            if "cubes" in function
            else function
        )
        LOGGER.info(
            f"{test_name} creation, {number} loops, best of "
            f"{len(time_vec)} runs: {results[function][0]:.3g}"
        )
    return globals()[cd["fun"]](setup_data)


def manual_timing(
    decimation_factor: int = 1,
    *,
    snapshot: str | None = None,
    creation_list: list[str] = None,
    search_list: list[str] = None,
    dry_run: bool = False,
    use_constant_number: int | None = None,
    number_balls: int | None = 100,
    dcrp: tuple[
        data_objects.MultiParticleDataset, list[NDArray], list[float], list[int]
    ]
    | None = None,
):
    if snapshot is None:
        snapshot = snapfile

    number_balls = 100 if number_balls is None else number_balls
    if number_balls < 1:
        raise ValueError("Number of search balls must be positive.")

    creation_dict, search_dict = get_creation_search_dicts()

    # default to all
    creation_list = (
        list(creation_dict.keys()) if creation_list is None else creation_list
    )

    # default to all
    search_list = list(search_dict.keys()) if search_list is None else search_list

    LOGGER.info("Running the following tests:")
    if creation_list:
        LOGGER.info(f"Creation:{creation_list}")
    if search_list:
        LOGGER.info(f"Search:{search_list}")

    ds, centers, radii, particle_numbers = (
        load_data(
            decimation_factor,
            filepath=snapshot,
            use_constant_number=use_constant_number,
            number_balls=number_balls,
        )
        if dcrp is None
        else dcrp
    )

    LOGGER.info("Beginning timing.")
    results = {}
    for test in search_list:
        sd = search_dict[test]
        creation_name = sd["tree"]
        search_obj = creation_dict[creation_name].get("search_obj")
        if not search_obj:
            need_timing = (
                results
                if creation_name in creation_list and creation_name not in results
                else None
            )
            LOGGER.debug(f"Generating {creation_name} search obj for {test} search")
            search_obj = get_search_obj(
                function=creation_name,
                dataset=ds,
                creation_dict=creation_dict,
                results=need_timing,
                dry_run=dry_run,
            )
            creation_dict[creation_name]["search_obj"] = search_obj
        globals()["data"] = ds.data_container
        globals()["search_obj"] = search_obj
        globals()["centers"] = centers
        globals()["radii"] = radii
        globals()["particle_numbers"] = particle_numbers
        scaling = sd.get("scaling", 1)
        if not dry_run:
            fun_str = sd["fun"].replace(
                "search_obj",
                "search_obj, centers=centers, radii=radii,"
                " particle_numbers=particle_numbers",
            )
            timer = timeit.Timer(
                fun_str, setup="import gc;gc.enable()", globals=globals()
            )
            number, time_vec = _run_timer(timer, add_scaling=scaling * len(radii))
        else:
            number = -1
            time_vec = [-1, -1] * second
        time_vec = _format_time(time_vec)
        results[test + "-search"] = (min(time_vec), time_vec)
        test_name = (test + f" ({cubes.nthreads} threads)") if "cubes" in test else test
        LOGGER.info(
            f"{test_name} search, {number} loops, best of "
            f"{len(time_vec)} runs: {results[test + '-search'][0]:.3g}"
        )

    LOGGER.debug("Running remaining creation tests")
    for test in creation_list:
        if test in results:
            continue
        get_search_obj(
            function=test,
            dataset=ds,
            creation_dict=creation_dict,
            results=results,
            dry_run=dry_run,
        )

    return results


# from https://stackoverflow.com/a/27434050
class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()

        # parse arguments in the file and store them in a blank namespace
        data = parser.parse_args(contents.split(), namespace=None)
        for k, v in vars(data).items():
            # set arguments in the target namespace if they have not been set yet
            if getattr(namespace, k, None) is None:
                setattr(namespace, k, v)


def parse_arguments(argv=None):
    if argv is None:
        # need to skip caller or it's picked up as the snapshot file
        argv = sys.argv[1:]

    description = """
    Time the various packingcubes creation and search methods. 
    
    Note that any methods specified as KDTree or kdtree refer to the 
    scipy.spatial KDTree module used for comparison, *not* a packingcubes
    construct.
    """
    epilog = """
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
        "--version",
        action="version",
        version=f"packingcubes: {packingcubes.__version__}",
    )
    parser.add_argument(
        "-d",
        "--decimation-factor",
        default=[1],
        help="The decimation interval (e.g. -d 10 specifies use every 10th particle)",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--loading-factor",
        help="""
        If the dataset is too large to load into RAM, you can provide an initial loading
        decimation factor. The --decimation-factor option will then be on top of this.
        So --loading-factor 10 --decimation-factor 10 100 is equivalent to
        --loading-factor 1 --decimation-factor 100 1000, but loads only 10 percent of
        the original data
        """,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--number-balls",
        help="Number of search balls to create. More balls = better statistics",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-s",
        "--number-search",
        help="""
        Use constant particle number search balls with the specified number of
        particles instead of constant volume search balls. This will be a
        slower startup!
        """,
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--dry",
        default=False,
        action="store_true",
        help="Skip actually running timing tests",
    )
    test_list_args = parser.add_argument_group(
        "Individual list", "Individual arguments to specify what tests to run"
    )
    creation_dict, search_dict = get_creation_search_dicts()
    for test in creation_dict:
        test_list_args.add_argument(
            f"--{test}-create",
            help=f"Create a {test} search object",
            dest="creation_list",
            action="append_const",
            const=test,
        )
    for test in search_dict:
        test_list_args.add_argument(
            f"--{test}-search",
            help=f"Search with a {test} search object",
            dest="search_list",
            action="append_const",
            const=test,
        )
    group_list_args = parser.add_argument_group(
        "Group list",
        (
            "Convenience groupings to specify a family of "
            "tests (like PackedTree creation and search)"
        ),
    )
    group_list_args.add_argument(
        "--pyoct",
        help="Create and search a PythonOctree object",
        dest="combined_list",
        action="append_const",
        const="pyoct",
    )
    group_list_args.add_argument(
        "--packed",
        help="Create and search a PackedTree object",
        dest="combined_list",
        action="append_const",
        const="packed",
    )
    group_list_args.add_argument(
        "--cubes",
        help="Create and search a Cubes object",
        dest="combined_list",
        action="append_const",
        const="cubes",
    )
    group_list_args.add_argument(
        "--kdtree",
        help="Create and search a Packed KDTree object",
        dest="combined_list",
        action="append_const",
        const="kdtree",
    )
    group_list_args.add_argument(
        "--scipy",
        help="Create and search a SciPy KDTree object",
        dest="combined_list",
        action="append_const",
        const="scipy",
    )
    conf_arg = parser.add_argument(
        "-c",
        "--config",
        help=(
            "Read in specified config file for arguments (CLI arguments will override)"
        ),
        type=open,
        action=LoadFromFile,
    )
    parser.add_argument(
        "snapshot", nargs="?", default=None, help="path to the snapshot file", type=str
    )
    parser.add_argument("--save", type=str, help="Text file to output results")

    args = parser.parse_args(argv)

    if args.verbose >= 2:
        loglvl = logging.DEBUG
    elif args.verbose >= 1:
        loglvl = logging.INFO
    else:
        loglvl = LOGGER.level
    LOGGER.setLevel(loglvl)
    return args


def _print_with_search_balls(
    *,
    outfile: TextIO,
    creation_list: list[str],
    search_list: list[str],
    decimation_factors: list[int],
    search_ball_sizes: list[int],
    results: dict[str, dict[str, tuple[unyt_quantity, unyt_array]]],
):
    print(f"m = {search_ball_sizes}", file=outfile)
    print("Creation times [s]:", file=outfile)
    for test in creation_list:
        result_array = np.full((len(decimation_factors),), np.nan)
        for i, df in enumerate(decimation_factors):
            res_name = f"df={df}_ns={search_ball_sizes[0]}"
            if res_name not in results:
                continue
            res = results[res_name][test][0]
            if res >= 0:
                result_array[i] = res.to("s")
        result_str = np.array2string(result_array, separator=", ", precision=3)
        print(f"{test} = {result_str}", file=outfile)
    print("Search times [ms]:", file=outfile)
    for test in search_list:
        result_array = np.full(
            (len(decimation_factors), len(search_ball_sizes)), np.nan
        )
        test_name = test + "-search"
        for i, df in enumerate(decimation_factors):
            for j, sb in enumerate(search_ball_sizes):
                res_name = f"df={df}_ns={sb}"
                if res_name not in results:
                    continue
                res = results[res_name][test_name][0]
                if res >= 0:
                    result_array[i, j] = res.to("ms")
        result_str = np.array2string(result_array, separator=", ", precision=4)
        print(f"{test} = {result_str}", file=outfile)


def _print_no_search_balls(
    *,
    outfile: TextIO,
    creation_list: list[str],
    search_list: list[str],
    decimation_factors: list[int],
    results: dict[str, dict[str, tuple[unyt_quantity, unyt_array]]],
):
    res_array_size = (len(decimation_factors),)
    print(f"n = {decimation_factors}", file=outfile)
    print("Creation times [s]:", file=outfile)
    for test in creation_list:
        result_array = np.full(res_array_size, np.nan)
        for i, df in enumerate(decimation_factors):
            key = f"df={df}"
            if key not in results:
                continue
            res = results[key][test][0]
            if res >= 0:
                result_array[i] = res.to("s")
        result_str = np.array2string(result_array, separator=", ", precision=3)
        print(f"{test} = {result_str}", file=outfile)
    print("Search times [ms]:", file=outfile)
    for test in search_list:
        result_array = np.full(res_array_size, np.nan)
        test_name = test + "-search"
        for i, df in enumerate(decimation_factors):
            key = f"df={df}"
            if key not in results:
                continue
            res = results[key][test_name][0]
            if res >= 0:
                result_array[i] = res.to("ms")
        result_str = np.array2string(result_array, separator=", ", precision=4)
        print(f"{test} = {result_str}", file=outfile)


def collate_results(
    *,
    creation_list: list[str],
    search_list: list[str],
    decimation_factors: list[int],
    search_ball_sizes: list[int],
    results: dict[str, dict[str, tuple[unyt_quantity, unyt_array]]],
    outfilepath: str,
):
    with open(outfilepath, "w") as outfile:
        if search_ball_sizes:
            _print_with_search_balls(
                outfile=outfile,
                creation_list=creation_list,
                search_list=search_list,
                decimation_factors=decimation_factors,
                search_ball_sizes=search_ball_sizes,
                results=results,
            )
        else:
            _print_no_search_balls(
                outfile=outfile,
                creation_list=creation_list,
                search_list=search_list,
                decimation_factors=decimation_factors,
                results=results,
            )


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig()
    LOGGER.info(f"Running with packingcubes v{packingcubes.__version__}")
    if args.dry:
        LOGGER.info("Dry run only. Actual timing will be skipped")
    creation_list = args.creation_list if args.creation_list else []
    search_list = args.search_list if args.search_list else []
    combined_list = args.combined_list if args.combined_list else []
    for t in combined_list:
        creation_list.append(t)
        search_list.append(t)
    results = {}
    ds_full, centers, radii, particle_numbers = load_data(
        1,
        filepath=args.snapshot if args.snapshot else snapfile,
        use_constant_number=None,
        number_balls=args.number_balls,
        loading_factor=args.loading_factor,
    )
    for df in args.decimation_factor:
        ds = set_decimation(ds=ds_full, decimation_factor=df)
        if args.number_search:
            for ns in args.number_search:
                centers, radii, particle_numbers = random_search_balls_constant_number(
                    ds, num_particles=ns, number_balls=args.number_balls
                )
                res_name = f"df={df}_ns={ns}"
                results[res_name] = manual_timing(
                    snapshot=args.snapshot,
                    decimation_factor=df,
                    creation_list=creation_list,
                    search_list=search_list,
                    dry_run=args.dry,
                    use_constant_number=ns,
                    number_balls=args.number_balls,
                    dcrp=(ds, centers, radii, particle_numbers),
                )
                if args.save:
                    collate_results(
                        creation_list=creation_list,
                        search_list=search_list,
                        decimation_factors=args.decimation_factor,
                        search_ball_sizes=args.number_search,
                        results=results,
                        outfilepath=args.save,
                    )
        else:
            results[f"df={df}"] = manual_timing(
                snapshot=args.snapshot,
                decimation_factor=df,
                creation_list=creation_list,
                search_list=search_list,
                dry_run=args.dry,
                use_constant_number=None,
                number_balls=args.number_balls,
                dcrp=(ds, centers, radii, particle_numbers),
            )
            if args.save:
                collate_results(
                    creation_list=creation_list,
                    search_list=search_list,
                    decimation_factors=args.decimation_factor,
                    search_ball_sizes=args.number_search,
                    results=results,
                    outfilepath=args.save,
                )
    print(results)  # noqa: T201
