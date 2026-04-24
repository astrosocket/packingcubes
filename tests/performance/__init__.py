"""Performance timing package

Collection of modules for timing performance of various elements of the
`packingcubes` library. The primary interfaces are the CLI interfaces in
timing.py and plot_timings.py, but programmatic access is also supported.

The remaining modules are organized by the element of the library they
are intended to test or the external comparison. Each module contains a
creation-type function whose only input is a Dataset and only output is a
object that can be searched with (e.g. a PackedTree). They also contain various
functions for searching with the object, with the signature
`object_name_search_type(search_object, **kwargs)`. These `kwargs` generally
include a list of sphere centers.


"""

from .plot_timings import (
    load_sim_results as load_sim_results,
)
from .plot_timings import (
    plot_expected_times as plot_expected_times,
)
from .plot_timings import (
    plot_normalized_times as plot_normalized_times,
)
from .plot_timings import (
    plot_parallel_scaling as plot_parallel_scaling,
)
from .plot_timings import (
    plot_raw_times as plot_raw_times,
)
from .timing import (
    collate_results as collate_results,
)
from .timing import (
    get_creation_search_dicts as get_creation_search_dicts,
)
from .timing import (
    get_data as get_data,
)
from .timing import (
    manual_timing as manual_timing,
)
from .timing import (
    save_results as save_results,
)

__all__ = [
    "manual_timing",
    "get_creation_search_dicts",
    "get_data",
    "collate_results",
    "save_results",
    "load_sim_results",
    "plot_raw_times",
    "plot_expected_times",
    "plot_normalized_times",
    "plot_parallel_scaling",
]
