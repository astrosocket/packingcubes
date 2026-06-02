"""Configuration for benchmarking"""

from collections.abc import Callable

from numpy.typing import NDArray

import packingcubes.data_objects as data_objects

from .brute import brute_force_creation, brute_force_search
from .cubes import (
    ParticleCubes,
    cubes_creation,
    cubes_get_particle_index_list_in_sphere,
    cubes_query_ball_points,
)
from .octree import python_octree_creation, python_octree_query_ball_point
from .optree import (
    OpTree,
    optree_creation,
    optree_query,
    optree_query_ball_point,
)
from .packed_tree import _NUM_REPS as _NUM_PACKED_QUERY_REPS
from .packed_tree import (
    PackedTree,
    packed_octree_creation,
    packed_octree_qbp_jitted,
    packed_octree_query,
    packed_octree_query_ball_point,
    packed_octree_query_ball_point_indices,
    packed_octree_query_jitted,
)
from .scipy import (
    KDTree,
    kdtree_creation,
    kdtree_query,
    kdtree_query_ball_point,
)

_DEFAULT_QUERY_SIZE = 100

# The NDArray is for the brute search creation method
type TSearchObj = PackedTree | ParticleCubes | OpTree | KDTree | NDArray
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
        "optree": optree_creation,
        "kdtree": kdtree_creation,
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
        "packnumbq": {
            "fun": packed_octree_query_jitted,  # needs dataset + tree
            "tree": "packed",
            "local": {"data": "data"},
            "scaling": _NUM_PACKED_QUERY_REPS,
            "config": {"k": _DEFAULT_QUERY_SIZE},
            "extended_description": f"""
            Returns k(={_DEFAULT_QUERY_SIZE})-closest particles as if run
            from jitted code, instead of normal python
            """,
        },
        "packq": {
            "fun": packed_octree_query,  # needs dataset + tree
            "tree": "packed",
            "local": {"data": "data"},
            "config": {"k": _DEFAULT_QUERY_SIZE},
            "extended_description": f"""
            Returns k(={_DEFAULT_QUERY_SIZE})-closest particles.
            """,
        },
        "cubes": {
            "fun": cubes_query_ball_points,
            "tree": "cubes",
        },
        "cubesli": {
            "fun": cubes_get_particle_index_list_in_sphere,
            "tree": "cubes",
            "config": {"strict": False},
        },
        "optree": {
            "fun": optree_query_ball_point,
            "tree": "optree",
        },
        "opq": {
            "fun": optree_query,
            "tree": "optree",
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
        "kdtree": {
            "fun": kdtree_query_ball_point,
            "tree": "kdtree",
        },
        "kdq": {
            "fun": kdtree_query,
            "tree": "kdtree",
            "config": {"k": _DEFAULT_QUERY_SIZE},
            "extended_description": f"""
            Returns k(={_DEFAULT_QUERY_SIZE})-closest particles.
            """,
        },
    }
    return creation_dict, search_dict
