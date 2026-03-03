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


def load_data(decimation_factor=10, *, name: str = simname, filepath: str = snapfile):
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
    random_search_balls(dataset)
    return dataset


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
    dataset = data_objects.InMemory(positions=np.array([0, 0, 0]))
    tree = packed_octree_creation(dataset)
    packed_octree_query_ball_point(tree)


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
    "kdtree": {"fun": "kdtree_creation", "precomp": False},
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
    "cubes": {
        "fun": "cubes_query_ball_points(search_obj)",
        "tree": "cubes",
        "precomp": True,
    },
    "kdtree": {
        "fun": "kdtree_query_ball_point(search_obj)",
        "tree": "kdtree",
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
    globals()["dataset"] = dataset
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


def check_precompile(*, creation_list: list[str], search_list: list[str]):
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
):
    if snapshot is None:
        snapshot = snapfile
    ds = load_data(decimation_factor, filepath=snapshot)
    LOGGER.info(
        f"Loaded {snapshot} with decimation factor {decimation_factor}"
        f"={len(ds)} particles"
    )

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

    check_precompile(creation_list=creation_list, search_list=search_list)

    LOGGER.info("Beginning timing.")
    results = {}
    for test in search_list:
        sd = search_dict[test]
        creation_name = sd["tree"]
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
        globals()["dataset"] = ds
        globals()["search_obj"] = search_obj
        if not dry_run:
            timer = timeit.Timer(sd["fun"], globals=globals())
            number, _ = timer.autorange()
            time_vec = timer.repeat(number=number) * second
            time_vec /= number  # chenage to per-loop
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
        "-d",
        "--decimation-factor",
        default=1,
        help="The decimation interval (e.g. -d 10 specifies use every 10th particle)",
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
        help="Create and search a KDTree object",
        dest="combined_list",
        action="append_const",
        const="kdtree",
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
    )
    print(results)  # noqa: T201
