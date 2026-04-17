import argparse
import contextlib
import json
import logging
import pickle
import sys
import time
from collections.abc import Callable
from functools import partial

import numpy as np
from numba import config, get_num_threads, set_num_threads
from numpy.typing import NDArray
from unyt import nanosecond, second, unyt_array, unyt_quantity

import packingcubes
import packingcubes.bounding_box as bbox
import packingcubes.data_objects as data_objects
import packingcubes.packed_tree as optree

from ._json_parsing import UnytEncoder
from .brute import brute_force_creation, brute_force_search
from .cubes import ParticleCubes, cubes_creation, cubes_query_ball_points
from .kdtree import (
    KDTree,
    packed_kdtree_creation,
    packed_kdtree_query,
    packed_kdtree_query_ball_point,
)
from .octree import python_octree_creation, python_octree_query_ball_point
from .packed_tree import _NUM_REPS as _NUM_PACKED_QUERY_REPS
from .packed_tree import (
    PackedTree,
    packed_octree_creation,
    packed_octree_qbp_jitted,
    packed_octree_query_ball_point,
    packed_octree_query_ball_point_indices,
)
from .scipy import (
    KDTree as SciTree,
)
from .scipy import (
    scipy_kdtree_creation,
    scipy_kdtree_query,
    scipy_kdtree_query_ball_point,
)

LOGGER = logging.getLogger(__name__)

particle_type = "PartType0"

rng = np.random.default_rng(0xBA55ADE89)
_DEFAULT_QUERY_SIZE = 100


def _get_random_data(random_size: int) -> data_objects.InMemory:
    return data_objects.InMemory(positions=rng.random((random_size, 3)) * 1000)


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


def _load_data(
    snapshot: str, loading_factor: int | None
) -> data_objects.GadgetishHDF5Dataset:
    LOGGER.debug("Beginning data loading")
    loading_pattern = _create_loading_pattern(snapshot, loading_factor)
    ds = data_objects.GadgetishHDF5Dataset(
        filepath=snapshot,
        data_slices=loading_pattern,
    )
    if ds.particle_type != particle_type:
        with contextlib.suppress(data_objects.DatasetError):
            ds.particle_type = particle_type
    loading_factor = loading_factor if loading_factor else 1
    LOGGER.info(
        f"Loaded {snapshot} with loading factor = "
        f"{loading_factor} => {len(ds):.3e} particles"
    )
    LOGGER.debug("Converting to InMemory")
    return data_objects.InMemory(positions=ds._positions)


def get_data(
    snapshot: str | None, *, loading_factor: int | None, random_size: int
) -> data_objects.MultiParticleDataset:
    return (
        _load_data(snapshot, loading_factor)
        if snapshot is not None
        else _get_random_data(random_size)
    )


def _process_data(
    *,
    dataset: data_objects.MultiParticleDataset,
    number_balls: int = 100,
) -> NDArray:
    box = dataset.bounding_box
    centers = []
    for _ in range(number_balls):
        centers.append(rng.random(box.size.size) * box.size + box.position)
    return centers


def set_decimation(
    *, ds: data_objects.Dataset, decimation_factor: int
) -> data_objects.Dataset:
    LOGGER.info(f"Decimating to {len(ds) / decimation_factor:.3e} particles")
    return data_objects.InMemory(
        positions=ds.positions.copy()[::decimation_factor, :],
    )


def _reset_data(ds: data_objects.Dataset) -> data_objects.Dataset:
    original_inds = np.argsort(ds.index)
    ds.reorder(original_inds)
    # verify
    assert np.all(ds.index[: len(ds) - 1] == ds.index[1:] - 1), (
        "Some indices were not reset!"
    )
    return ds


def random_search_balls(
    ds: data_objects.Dataset,
    *,
    centers: NDArray,
    num_particles: int = 1000,
    error_threshold: float = 0.1,
    iterations_threshold: int = 20,
) -> tuple[list[float], list[int]]:
    LOGGER.info(
        f"Generating constant particle number search balls of size {num_particles}"
    )

    if num_particles > len(ds):
        # need to return a radii bigger than box -> just use twice max dx
        # Technically only need sqrt(3) * max dx, but there's no penalty in
        # having the box be larger
        r = 2 * max(ds.bounding_box.size)
        radii = np.full((len(centers),), r).tolist()
        particle_numbers = np.full((len(centers),), len(ds)).tolist()
        LOGGER.debug(
            f"""
            Requested {num_particles} particles but data only has {len(ds)}.
            Filling with {r=} and {len(ds)} particles.
            """
        )
        return radii, particle_numbers

    kdtree = optree.KDTree(data=ds.positions, copy_data=True)
    box = ds.bounding_box
    bad_balls = 0
    radii = []
    particle_numbers = []
    for i, center in enumerate(centers):
        # bisection method:
        # assume r = 0 => n_enclosed = 0
        # bisect radii until |n_enclosed - num_particles|/num_particles < 5%
        # Could use query(k=num_particles) instead, but likely slower for big
        # num_particles

        # start with min dx/10
        r = min(ds.bounding_box.size) / 10
        LOGGER.debug(f"Starting search for ball {i} with center {center} and r0={r}")
        n_enclosed = len(
            kdtree.query_ball_point(
                center,
                r,
                strict=True,
                return_data_indices=True,
                return_sorted=False,
                return_lists=False,
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
                    strict=True,
                    return_data_indices=True,
                    return_sorted=False,
                    return_lists=False,
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
                    strict=True,
                    return_data_indices=True,
                    return_sorted=False,
                    return_lists=False,
                )
            )
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
            LOGGER.info(f"Taking too long on sphere {i}")
            bad_balls += 1
        radii.append(r)
        particle_numbers.append(n_enclosed)
    if bad_balls:
        LOGGER.debug(f"Iteration threshold reached on {bad_balls} spheres")
    if bad_balls > len(centers) / 10:
        LOGGER.warning(
            f"More than 10% of spheres ({bad_balls}) crossed the iteration threshold."
            " Timing stats may suffer"
        )
    return radii, particle_numbers


def _remove_problem_classes_from_state(self):
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
    ds = _process_data(decimation_factor=decimation_factor)
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
                    node.__getstate__ = partial(
                        _remove_problem_classes_from_state, node
                    )
            case "packed":
                tree.__getstate__ = partial(_remove_problem_classes_from_state, tree)
        b = pickle.dumps(tree)
        print(f"{name}: {len(b)}")  # noqa


# The NDArray is for the brute search creation method
type TSearchObj = PackedTree | ParticleCubes | KDTree | SciTree | NDArray
""" A search object """


def get_creation_search_dicts() -> tuple[
    dict[str, Callable[[data_objects.Dataset], TSearchObj]], dict
]:
    """Get functions to be timed and any additional config options

    For the search dictionary, the "fun" field is the function to be timed,
    the "tree" field is the search object needed, the "config" field contains
    any configuration values known ahead of time, and the "local" field contains
    variables that need to be resolved in the local namespace. The scaling field
    is for additional scaling of the time taken to account for internal loops
    """
    # in the future we should generate these dynamically
    creation_dict = {
        "pyoct": python_octree_creation,
        "packed": packed_octree_creation,
        "cubes": cubes_creation,
        "kdtree": packed_kdtree_creation,
        "scipy": scipy_kdtree_creation,
        "brute": brute_force_creation,
    }
    search_dict = {
        "pyoct": {
            "fun": python_octree_query_ball_point,
            "tree": "pyoct",
        },
        "packed": {
            "fun": packed_octree_query_ball_point,
            "tree": "packed",
        },
        "packli": {
            "fun": packed_octree_query_ball_point_indices,  # needs dataset + tree
            "tree": "packed",
            "local": {"data": "data"},
            "extended_description": """
            Returns the list of indices, equivalent to KDTree.query_ball_point()
            """,
        },
        "packnumb": {
            "fun": packed_octree_qbp_jitted,
            "tree": "packed",
            "scaling": _NUM_PACKED_QUERY_REPS,
            "extended_description": """
            Compute timing as if run from jitted code instead of normal python.
            """,
        },
        "cubes": {
            "fun": cubes_query_ball_points,
            "tree": "cubes",
        },
        "kdtree": {
            "fun": packed_kdtree_query_ball_point,
            "tree": "kdtree",
        },
        "kdq": {
            "fun": packed_kdtree_query,
            "tree": "kdtree",
            "config": {"k": _DEFAULT_QUERY_SIZE},
            "extended_description": f"""
            Returns k(={_DEFAULT_QUERY_SIZE})-closest particles, equivalent to 
            KDTree.query().
            """,
        },
        "brute": {
            "fun": brute_force_search,
            "tree": "brute",
        },
        "scipy": {
            "fun": scipy_kdtree_query_ball_point,
            "tree": "scipy",
        },
        "sciq": {
            "fun": scipy_kdtree_query,
            "tree": "scipy",
            "config": {"k": _DEFAULT_QUERY_SIZE},
            "extended_description": f"""
            Returns k(={_DEFAULT_QUERY_SIZE})-closest particles.
            """,
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


type TResult = tuple[unyt_quantity, unyt_array]
""" A result tuple in the form (min(time_vector), time_vector) """


def _run_creation_perf_timer(
    creation_fun: Callable[[data_objects.Dataset], TSearchObj],
    dataset: data_objects.MultiParticleDataset,
) -> TResult:
    """Time search object creation using perf_timer"""
    # do initial run to catch any precompilation issues and to prime data loading
    creation_fun(dataset)
    # reset data
    _reset_data(dataset)

    def timeit(
        creation_fun,
        dataset: data_objects.MultiParticleDataset,
        number: int,
    ) -> unyt_quantity:
        t0 = time.perf_counter_ns()
        delta = 0
        for _ in range(number):
            t1 = time.perf_counter_ns()
            _reset_data(dataset)
            delta += time.perf_counter_ns() - t1
            creation_fun(dataset)
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
                number=number,
            ).to("s")
    except ValueError as ve:
        LOGGER.warning(ve)
        return -1, [-1, -1] * second
    return number, time_vec


def _run_search_perf_timer(
    *, search_dict: dict, search_obj, dry_run: bool = False, **kwargs
) -> TResult:
    """Time search object creation using perf_timer"""
    if dry_run:
        return -1, [-1, -1] * second

    search_fun = search_dict["fun"]
    config_kwargs = search_dict.get("config", {})

    # do initial run to catch any precompilation issues and to prime data loading
    search_fun(search_obj, **config_kwargs, **kwargs)

    def timeit(*, search_fun, search_obj, number: int, **kwargs) -> unyt_quantity:
        t0 = time.perf_counter_ns()
        for _ in range(number):
            search_fun(search_obj, **kwargs)
        tfinal = time.perf_counter_ns()
        return (tfinal - t0) * nanosecond

    # do manual autorange
    scale = 1
    done = False
    try:
        while not done:
            for i in [1, 2, 5]:
                number = i * scale
                time_taken = timeit(
                    search_fun=search_fun,
                    search_obj=search_obj,
                    number=number,
                    **config_kwargs,
                    **kwargs,
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
                search_fun=search_fun,
                search_obj=search_obj,
                number=number,
                **config_kwargs,
                **kwargs,
            ).to("s")
    except ValueError as ve:
        LOGGER.warning(ve)
        return -1, [-1, -1] * second
    return number, time_vec


def _process_time_vec(
    *, time_vec: unyt_array, number: int, extra_scaling: int = 1
) -> TResult:
    """Scale and find minimum of time vector."""
    time_vec /= number
    time_vec /= extra_scaling
    time_vec = _format_time(time_vec)
    return (min(time_vec), time_vec)


def get_search_obj(
    *,
    name: str,
    dataset: data_objects.Dataset,
    creation_fun: Callable[[data_objects.Dataset], TSearchObj],
    results: dict = None,
    dry_run: bool = False,
) -> TSearchObj:
    if results is not None:
        LOGGER.debug(f"Timing {name} creation")
        if not dry_run:
            number, time_vec = _run_creation_perf_timer(
                creation_fun=creation_fun,
                dataset=dataset,
            )
        else:
            number = -1
            time_vec = [-1, -1] * second
        results[name] = _process_time_vec(
            time_vec=time_vec, number=number, extra_scaling=1
        )
        test_name = (
            (name + f" ({get_num_threads()} threads)")
            if "cubes" in name or "kdtree" in name
            else name
        )
        LOGGER.info(
            f"{test_name} creation, {number} loops, best of "
            f"{len(time_vec)} runs: {results[name][0]:.3g}"
        )

    so = creation_fun(dataset)
    so.data_container = dataset.data_container
    return so


def manual_timing(
    *,
    ds: data_objects.MultiParticleDataset,
    centers: list[NDArray],
    radii: list[float],
    particle_numbers: list[int],
    creation_list: list[str],
    search_list: list[str],
    dry_run: bool = False,
    creation_cache: dict | None = None,
    number_threads: int | None = None,
) -> dict:
    number_threads = (
        config.NUMBA_NUM_THREADS if number_threads is None else number_threads
    )
    if number_threads < 1 or config.NUMBA_NUM_THREADS < number_threads:
        raise ValueError(
            f"""
            Invalid number of threads specified. Must be between 1 and 
            {config.NUMBA_NUM_THREADS}.
            """
        )

    set_num_threads(number_threads)

    creation_cache = {} if creation_cache is None else creation_cache

    creation_dict, search_dict = get_creation_search_dicts()

    LOGGER.info("Running the following tests:")
    if creation_list:
        LOGGER.info(f"Creation:{creation_list}")
    if search_list:
        LOGGER.info(f"Search:{search_list}")

    LOGGER.info("Beginning timing.")
    results = {}
    for test in search_list:
        sd = search_dict[test]
        creation_name = sd["tree"]
        search_obj = creation_cache.get(creation_name)
        if not search_obj:
            need_timing = (
                results
                if creation_name in creation_list and creation_name not in results
                else None
            )
            LOGGER.debug(f"Generating {creation_name} search obj for {test} search")
            search_obj = get_search_obj(
                name=creation_name,
                dataset=ds,
                creation_fun=creation_dict[creation_name],
                results=need_timing,
                dry_run=dry_run,
            )
            creation_cache[creation_name] = search_obj

        number, time_vec = _run_search_perf_timer(
            search_dict=sd,
            search_obj=search_obj,
            dry_run=dry_run,
            dataset=ds,
            data=ds.data_container,
            centers=centers,
            radii=radii,
            particle_numbers=particle_numbers,
        )

        # need to scale by number of balls
        extra_scaling = len(centers) * sd.get("scaling", 1)
        results[test + "-search"] = _process_time_vec(
            time_vec=time_vec,
            number=number,
            extra_scaling=extra_scaling,
        )
        test_name = (
            (test + f" ({get_num_threads()} threads)")
            if "cubes" in test or "kdtree" in test
            else test
        )
        LOGGER.info(
            f"{test_name} search, {number} loops, best of "
            f"{len(time_vec)} runs: {results[test + '-search'][0]:.3g}"
        )

    LOGGER.debug("Running remaining creation tests")
    for test in creation_list:
        if test in results or test in creation_cache:
            continue
        get_search_obj(
            name=test,
            dataset=ds,
            creation_dict=creation_dict,
            results=results,
            dry_run=dry_run,
        )

    return results


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
        help="""
        The decimation interval (e.g. -d 10 specifies use every 10th particle).
        Can be provided as a list
        """,
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
        the original data. Data is loaded in "fat chunks" so --loading-factor 20
        will load ten chunks of size len(data)/200.
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
        Number of particles in a search ball. Can be provided as a list
        """,
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "-t",
        "--number-threads",
        default=[get_num_threads()],
        help=f"""
        For those tests which can benefit from parallelization (cubes, KDTree),
        specify the number of threads to run on (e.g. for use in scaling 
        benchmarks). Note that this number must be less than the maximum number
        of threads available ({config.NUMBA_NUM_THREADS}). Can be provided as
        a list.
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
    for test, test_prop in search_dict.items():
        test_list_args.add_argument(
            f"--{test}-search",
            help=(
                f"Search with a {test_prop['tree']} search object. "
                + test_prop.get("extended_description", "")
            ),
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

    group_list_const = ["pyoct", "packed", "cubes", "kdtree", "scipy"]
    group_list_names = [
        "PythonOctree",
        "PackedTree",
        "Cubes",
        "Packed KDTree",
        "SciPy KDTree",
    ]
    for const, name in zip(group_list_const, group_list_names, strict=True):
        group_list_args.add_argument(
            f"--{const}",
            help=f"Create and search a {name} object",
            dest="combined_list",
            action="append_const",
            const=const,
        )

    group_list_args.add_argument(
        "--all",
        help="Run all tests",
        dest="combined_list",
        action="append_const",
        const="all",
    )
    group_list_args.add_argument(
        "--most",
        help="""
        Run most of the tests. Specifically, this is the equivalent of 
        `--packed --kdtree --scipy --kdq-search --sciq-search`
        """,
        dest="combined_list",
        action="append_const",
        const="most",
    )
    conf_arg = parser.add_argument(
        "-c",
        "--config",
        help="""
        Read in specified config file for arguments (CLI arguments will override)
        """,
        type=open,
        action=_LoadFromFile,
    )
    parser.add_argument(
        "-r",
        "--random-size",
        help="""
        Set the amount of randomly generated data. Default is 100 million (10^8)
        particles.
        """,
        default=100_000_000,
        type=int,
    )
    parser.add_argument(
        "snapshot",
        nargs="?",
        default=None,
        help="""
        Path to the snapshot file. If not provided, data will be randomly generated
        """,
        type=str,
    )
    parser.add_argument("--save", type=str, help="Text file to output results")

    args = parser.parse_args(argv)

    if args.combined_list and "all" in args.combined_list:
        args.combined_list.remove("all")
        args.creation_list = [] if args.creation_list is None else args.creation_list
        args.search_list = [] if args.search_list is None else args.search_list
        args.creation_list.extend(creation_dict)
        args.search_list.extend(search_dict)

    if args.combined_list and "most" in args.combined_list:
        args.combined_list.remove("most")
        args.combined_list.extend(["packed", "kdtree", "scipy"])
        args.search_list = [] if args.search_list is None else args.search_list
        args.search_list.extend(["kdq", "sciq"])

    if args.verbose >= 2:
        loglvl = logging.DEBUG
    elif args.verbose >= 1:
        loglvl = logging.INFO
    else:
        loglvl = LOGGER.level
    LOGGER.setLevel(loglvl)
    return args


def _parse_outcome_as_index(outcome: str, **kwargs):
    """
    Convert outcome strings to indices
    """
    # split into string list and drop 'out'
    pieces = outcome.split("_")[1:]
    index_list = []
    for piece in pieces:
        name, value = piece.split("=")
        index_dict = kwargs.get(f"{name}_dict")
        if index_dict is None:
            index_list.append([])
            continue
        value = int(value)
        index_list.append(index_dict[value])
    return tuple(index_list)


def _add_or_create_array(
    *, dict_: dict, name: str, idx, size, value: unyt_quantity | None
):
    """
    Add element to matrix, creating it if it doesn't exist
    """
    if value is None:
        return
    with contextlib.suppress(TypeError):
        value = value[0]
    if name not in dict_:
        dict_[name] = np.full(size, np.nan) * value.units
    dict_[name][idx] = value


def collate_results(
    *,
    results: dict,
) -> dict:
    """
    Reorder results dictionary such that creation/search-names/types are top-level keys
    """
    creation_list = list(results.get("creation-tests", []))
    search_list = [t + "-search" for t in results.get("search-tests", [])]
    test_list = creation_list + search_list
    outcomes = [o for o in results if "out" in o]

    collated = {}
    decimations = results.get("decimations", [])
    ball_sizes = results.get("ball_sizes", [])
    number_threads = results.get("number_threads", [])
    collated["decimations"] = decimations
    collated["m"] = ball_sizes
    collated["num_threads"] = number_threads
    df_dict = {k: i for i, k in enumerate(decimations)}
    sb_dict = {k: i for i, k in enumerate(ball_sizes)}
    threads_dict = {k: i for i, k in enumerate(number_threads)}

    collated_creation = {}
    collated_threads = {}
    collated_search = {}

    for outcome in outcomes:
        idx = _parse_outcome_as_index(
            outcome,
            df_dict=df_dict,
            sb_dict=sb_dict,
            threads_dict=threads_dict,
        )
        res = results[outcome]

        for creation in creation_list:
            _add_or_create_array(
                dict_=collated_creation,
                name=creation,
                idx=idx[0],
                size=(len(df_dict),),
                value=res.get(creation),
            )
        for test in test_list:
            _add_or_create_array(
                dict_=collated_threads,
                name=test,
                idx=idx[2],
                size=(len(threads_dict),),
                value=res.get(test),
            )
        for search in search_list:
            _add_or_create_array(
                dict_=collated_search,
                name=search,
                idx=idx[:2],
                size=(len(df_dict), len(sb_dict)),
                value=res.get(search),
            )

    collated["creation"] = collated_creation
    collated["threads"] = collated_threads
    collated["search"] = collated_search

    name_list = ["creation", "threads", "search"]
    dict_list = [
        collated_creation,
        collated_threads,
        collated_search,
    ]
    for test in test_list:
        collated[test] = {}
        for name, dict_ in zip(name_list, dict_list, strict=True):
            if test in dict_:
                collated[test][name] = dict_[test]

    return collated


def save_results(
    *,
    results: dict,
    snapshot_info: dict,
    outfilepath: str,
    raw_results: dict | None = None,
):
    if raw_results is not None:
        results["raw"] = raw_results
    results["snapshot_info"] = snapshot_info
    with open(outfilepath, "w") as outfile:
        json.dump(
            results,
            outfile,
            allow_nan=True,
            sort_keys=True,
            cls=UnytEncoder,
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
    # ensure no duplicate tests
    creation_list = set(creation_list)
    search_list = set(search_list)

    results = {}
    if creation_list:
        results["creation-tests"] = creation_list
    if search_list:
        results["search-tests"] = search_list
    results["decimations"] = args.decimation_factor
    if args.number_search:
        results["ball_sizes"] = args.number_search
    results["number_threads"] = args.number_threads

    ds_full = get_data(
        args.snapshot, loading_factor=args.loading_factor, random_size=args.random_size
    )

    snapshot_info = {
        "n": len(ds_full),
        "name": "random" if args.snapshot is None else args.snapshot,
        "particle_type": particle_type,
    }

    centers = _process_data(
        dataset=ds_full,
        number_balls=args.number_balls,
    )
    for df in args.decimation_factor:
        ds = set_decimation(ds=ds_full, decimation_factor=df)
        for num_threads in args.number_threads:
            creation_cache = {}
            for ns in args.number_search:
                radii, particle_numbers = random_search_balls(
                    ds, num_particles=ns, centers=centers
                )
                res_name = f"out_df={df}_sb={ns}_threads={num_threads}"
                results[res_name] = manual_timing(
                    ds=ds,
                    centers=centers,
                    radii=radii,
                    particle_numbers=particle_numbers,
                    creation_list=creation_list,
                    search_list=search_list,
                    dry_run=args.dry,
                    creation_cache=creation_cache,
                    number_threads=num_threads,
                )
                # update results on each search ball size, since this can be slow
                if args.save:
                    save_results(
                        results=collate_results(results=results),
                        outfilepath=args.save,
                        snapshot_info=snapshot_info,
                    )
    print(collate_results(results=results))  # noqa: T201
