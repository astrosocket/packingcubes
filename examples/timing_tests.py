import argparse
import contextlib
import logging
import pickle
import sys
import timeit
from functools import partial

import numpy as np
import yt
from scipy.spatial import KDTree
from unyt import second, unyt_array, unyt_quantity
from yt.units import Msun, kiloparsec

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

centers = []
radii = []
particle_numbers = []


def load_data(
    decimation_factor=10,
    *,
    name: str = simname,
    filepath: str = snapfile,
    use_constant_number: int | None = None,
    number_balls: int = 100,
):
    ds = data_objects.GadgetishHDF5Dataset(name=name, filepath=filepath)
    # ds._positions = ds._positions[: int(len(ds) / decimation_factor), :]
    if ds.particle_type != particle_type:
        with contextlib.suppress(data_objects.DatasetError):
            ds.particle_type = particle_type
    # Bug fix: convert to InMemory version so that when testing cubing versions
    # we don't try to reload the dataset (because cubing does every particle
    # type available by setting the particle type, which reloads the data,
    # undoing the decimation here. This means we don't need to manually set the
    # index or bounding box either, which is a nice benefit
    dataset = data_objects.InMemory(
        positions=ds._positions[:: int(decimation_factor), :]
    )
    if use_constant_number:
        if use_constant_number < 10:
            LOGGER.warning(
                "Specifying search balls with fewer than 10 particles."
                " I hope you're sure!"
            )
        random_search_balls_constant_number(
            dataset, number_balls=number_balls, num_particles=use_constant_number
        )
    else:
        random_search_balls_constant_volume(dataset, number_balls=number_balls)

    return dataset


def reset_data(ds: data_objects.Dataset) -> data_objects.Dataset:
    original_inds = np.argsort(ds.index)
    ds.reorder(original_inds)
    return ds


def random_search_balls_constant_volume(ds, *, number_balls: int = 100):
    box = ds.bounding_box
    for _ in range(number_balls):
        centers.append(rng.random(box.size.size) * box.size + box.position)
        radii.append(10 ** (rng.random() * np.log10(rng.choice(box.size))))


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
    kdtree = optree.KDTree(data=ds.positions)
    # kdtree = KDTree(data=ds.positions, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD)
    box = ds.bounding_box
    bad_balls = 0
    for i in range(number_balls):
        # bisection method:
        # keep center generation and use radius generation for initial radii
        # assume r = 0 => n_enclosed = 0
        # bisect radii until |n_enclosed - num_particles|/num_particles < 5%
        center = rng.random(box.size.size) * box.size + box.position
        r = 10 ** (rng.random() * np.log10(rng.choice(box.size)))
        LOGGER.debug(f"Starting search for ball {i} with center {center} and r0={r}")
        n_enclosed = len(
            kdtree.query_ball_point(
                center,
                r,
                strict=True,
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
                )
            )
        rupper = r
        LOGGER.debug(
            f"Starting refined search with r1={rlower:.4g} and rupper={rupper:.4g}"
            f" and {n_enclosed} particles"
        )
        num_iterations = 0
        error = abs(n_enclosed - num_particles) / num_particles
        while num_iterations < iterations_threshold and error >= error_threshold:
            r = (rupper + rlower) / 2
            LOGGER.debug(
                f"Have {n_enclosed} particles ({error=:.3g}). Checking {r=:.4g}"
            )
            n_enclosed = len(kdtree.query_ball_point(center, r, strict=True))
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


def packed_octree_query_ball_point_indices(
    data: data_objects.DataContainer, tree: optree.PackedTree
):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_index_list_in_sphere(
            data=data,
            center=c,
            radius=r,
            strict=False,
        )


def packed_kdtree_creation(ds):
    return optree.KDTree(data=ds.positions, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD)


def packed_kdtree_query_ball_point(
    tree: optree.KDTree,
    *,
    particle_numbers: list[int] = particle_numbers,
):
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        sph_inds = tree.query_ball_point(x=c, r=r, strict=True, return_lists=False)
        if particle_numbers and len(sph_inds) != particle_numbers[i]:
            raise ValueError(
                f"""
                Particle number mismatch: expected {particle_numbers[i]} particles
                for ball {i} and only got {len(sph_inds)}.
                """
            )


# we want the PackedTree stuff to be pre-compiled
def precompile():
    dataset = data_objects.InMemory(positions=np.array([0, 0, 0]))
    tree = packed_octree_creation(dataset)
    packed_octree_query_ball_point(tree)
    packed_octree_query_ball_point_indices(dataset, tree)


def scipy_kdtree_creation(ds):
    return KDTree(data=ds.positions, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD)


def scipy_kdtree_query_ball_point(
    tree: KDTree,
    particle_numbers: list[int] = particle_numbers,
):
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        sph_inds = tree.query_ball_point(x=c, r=r)
        if particle_numbers and len(sph_inds) != particle_numbers[i]:
            raise ValueError(
                f"""
                Particle number mismatch: expected {particle_numbers[i]} particles
                for ball {i} and only got {len(sph_inds)}.
                """
            )


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
    # InMemory datasets only are PartType0 by default
    args = cubes._process_args(["-t0", "--", str(dataset.filepath)])
    box = cubes._process_box(dataset=dataset, args=args)
    cubes_query_ball_points(cubing((dataset, args, box)))
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
            particle_types=particle_type,
            center=c,
            radius=r,
        )


creation_dict = {
    "pyoct": {
        "fun": "python_octree_creation",
        "precomp": True,
    },
    "packed": {"fun": "packed_octree_creation", "precomp": True},
    "cubes": {
        "fun": "cubing",
        "setup": cubing_setup,
        "precomp": True,
    },
    "kdtree": {"fun": "packed_kdtree_creation", "precomp": True},
    "scipy": {"fun": "scipy_kdtree_creation", "precomp": False},
    # skipping yt creation for now
}
search_dict = {
    "pyoct": {
        "fun": "python_octree_query_ball_point(search_obj)",
        "tree": "pyoct",
        "precomp": False,
    },
    "packed": {
        "fun": "packed_octree_query_ball_point(search_obj)",
        "tree": "packed",
        "precomp": True,
    },
    "packli": {
        "fun": (
            "packed_octree_query_ball_point_indices(data, search_obj)"
        ),  # needs dataset + tree
        "tree": "packed",
        "precomp": True,
    },
    "cubes": {
        "fun": "cubes_query_ball_points(search_obj)",
        "tree": "cubes",
        "precomp": True,
    },
    "kdtree": {
        "fun": "packed_kdtree_query_ball_point(search_obj)",
        "tree": "kdtree",
        "precomp": True,
    },
    "scipy": {
        "fun": "scipy_kdtree_query_ball_point(search_obj)",
        "tree": "scipy",
        "precomp": False,
    },
    # skipping yt for now
}


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


def get_search_obj(
    *,
    function: str,
    dataset: data_objects.Dataset,
    results: dict = None,
    dry_run: bool = False,
):
    cd = creation_dict[function]
    setup = cd.get("setup", lambda dataset: dataset)
    setup_data = setup(dataset=dataset)
    globals()["data"] = dataset.data_container
    globals()["setup_data"] = setup_data

    if results is not None:
        LOGGER.debug(f"Timing {function} creation")
        statement = f"reset_data(dataset);{cd['fun']}(setup_data)"
        if not dry_run:
            timer = timeit.Timer(statement, globals=globals())
            number, _ = timer.autorange()
            time_vec = timer.repeat(number=number) * second
            time_vec /= number
        else:
            number = -1
            time_vec = [-1, -1] * second
        time_vec = _format_time(time_vec)
        results[function] = (min(time_vec), time_vec)
        LOGGER.info(
            f"{function} creation, {number} loops, best of "
            f"{len(time_vec)} runs: {results[function][0]:.3g}"
        )
    return globals()[cd["fun"]](setup_data)


def check_precompile(
    *,
    creation_list: list[str],
    search_list: list[str],
    use_constant_number: int | None = None,
):
    if use_constant_number:
        LOGGER.info("Constant number spheres require precompiling")
        precompile()
        LOGGER.debug("Finished precompiling")
        return
    for test in creation_list:
        if creation_dict[test]["precomp"]:
            LOGGER.info(f"{test}-creation requires precompiling")
            precompile()
            LOGGER.debug("Finished precompiling")
            return
    for test in search_list:
        if search_dict[test]["precomp"]:
            LOGGER.info(f"{test}-search requires precompiling")
            precompile()
            LOGGER.debug("Finished precompiling")
            return
    return


def manual_timing(
    decimation_factor: int = 1,
    *,
    snapshot: str | None = None,
    creation_list: list[str] = None,
    search_list: list[str] = None,
    dry_run: bool = False,
    use_constant_number: int | None = None,
    number_balls: int | None = 100,
):
    if snapshot is None:
        snapshot = snapfile

    number_balls = 100 if number_balls is None else number_balls
    if number_balls < 1:
        raise ValueError("Number of search balls must be positive.")

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

    check_precompile(
        creation_list=creation_list,
        search_list=search_list,
        use_constant_number=use_constant_number,
    )

    LOGGER.debug("Beginning data loading")
    ds = load_data(
        decimation_factor,
        filepath=snapshot,
        use_constant_number=use_constant_number,
        number_balls=number_balls,
    )
    LOGGER.info(
        f"Loaded {snapshot} with decimation factor {decimation_factor}"
        f"={len(ds):.3e} particles"
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
                results=need_timing,
                dry_run=dry_run,
            )
            creation_dict[creation_name]["search_obj"] = search_obj
        globals()["dataset"] = ds
        globals()["search_obj"] = search_obj
        if not dry_run:
            timer = timeit.Timer(sd["fun"], globals=globals())
            number, _ = timer.autorange()
            time_vec = timer.repeat(number=number) * second
            time_vec /= number  # change to per-loop
            time_vec /= len(centers)  # change to per-sphere (average)
        else:
            number = -1
            time_vec = [-1, -1] * second
        time_vec = _format_time(time_vec)
        results[test] = (min(time_vec), time_vec)
        LOGGER.info(
            f"{test} search, {number} loops, best of "
            f"{len(time_vec)} runs: {results[test][0]:.3g}"
        )

    LOGGER.debug("Running remaining creation tests")
    for test in creation_list:
        if test in results:
            continue
        get_search_obj(test, dataset=ds, results=results, dry_run=dry_run)

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
        default=1,
        help="The decimation interval (e.g. -d 10 specifies use every 10th particle)",
        type=int,
    )
    parser.add_argument(
        "-n",
        "--number-balls",
        help="Number of search balls to create. More balls = better statistics",
        type=int,
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

    args = parser.parse_args(argv)

    if args.verbose >= 2:
        loglvl = logging.DEBUG
    elif args.verbose >= 1:
        loglvl = logging.INFO
    else:
        loglvl = LOGGER.level
    LOGGER.setLevel(loglvl)
    return args


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
    results = manual_timing(
        snapshot=args.snapshot,
        decimation_factor=args.decimation_factor,
        creation_list=creation_list,
        search_list=search_list,
        dry_run=args.dry,
        use_constant_number=args.number_search,
        number_balls=args.number_balls,
    )
    print(results)  # noqa: T201
