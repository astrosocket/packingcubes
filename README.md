# packingcubes

[![PyPI](https://img.shields.io/pypi/v/packingcubes)](https://pypi.org/project/packingcubes)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/packingcubes)](https://pypi.org/project/packingcubes/)

<!--- Tests and style --->
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)


Compact octree implementation used for Socket

`packingcubes` aims to provide a fast, minimal-memory-usage octree
implementation, specialized for use in astronomical/astrophysical contexts.
It's written in pure python, with [Numba](https://numba.pydata.org/)-based
acceleration of the critical code paths.

## Requirements
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/packingcubes)](https://pypi.org/project/packingcubes/) 
is required. Python versions outside this range may work, but their usage is
not supported, and at least some features are >Python 3.12. 

### Python packages
* `numpy` - required for core routines
* `numba` - required for core routines (and most of the speed)
* `h5py` - required for reading snapshot data and saving `Cubes` and
   `PackedTree`s
* `xxhash` - required for core routines (packed metadata format)

### Optional packages
Visualization (the `viz` group):

* `matplotlib` - to use basic octree visualization and plotting (see
   Basic_Usage in the Examples)
* `pygfx` - to do interactive octree visualization (see Example_PackedTree in
   the Examples)
* `rendercanvas` - to do interactive octree visualization
* `pyside6` - performant interactive octree visualization (uses Qt, other
   options include `wgpu-py`, `pyodide`. See the
  [rendercanvas backends](https://rendercanvas.readthedocs.io/stable/backends.html)
   documentation for more details

Jupyter (the `jupyter` group):

* `jupyter-rfb` - for interactive octree visualization in a notebook (see the Visualization section, above)

Benchmark (the `benchmark` group):

* `scipy` - we benchmark against `scipy`'s `KDTree`
* `unyt` - for unit-aware timing purposes

The `all` group combines all of the above.

## Basic Usage
### Installation
We're on [PyPI](https://pypi.org/project/packingcubes), so installation is as
simple as

```bash
pip install packingcubes
```
or
```bash
uv pip install packingcubes
```
or
```bash
pixi add packingcubes --pypi
```

Additional package requirements can be installed via optional dependencies (see
the requirements section for the lists). Examples:
```sh
pip install "packingcubes[viz, jupyter]"
```
or 
```sh
pixi add packingcubes[all] --pypi
```


### Construction
#### From a snapshot on disk
From the command line:

```sh
packcubes SNAPSHOT OUTPUT
```
will generate a `Cubes` data structure and store it in the `OUTPUT` hdf5 file
along with the sorted positions and shuffle-list.

Alternatively:
```python
import packingcubes

cubes = packingcubes.Cubes("path/to/snapshot.hdf5")

cubes.save("path/to/output.hdf5")
# sorted positions/indices are stored in .snapshot_sorted.hdf5 by default

# to save elsewhere, use
dataset = packingcubes.HDF5Dataset(
    "path/to/snapshot.hdf5", sorted_filepath="path/to/output.hdf5"
)
cubes = packingcubes.Cubes(dataset)
```

#### From positions in memory
If you already have `positions_data` as an Nx3 matrix, you can use

``` python
cubes = packingcubes.Cubes(positions_data)
```

Note: this data will be sorted in place! You may want to make a copy first.

You can also do the following for easy saving. 
The data will still not be copied (by default)

``` python
dataset = packingcubes.InMemory(positions=positions_data)
cubes = packingcubes.Cubes(dataset)
dataset.save("path/to/output.hdf5")
```

Several configuration options are available, see `packcubes --help` or `help(packingcubes.Cubes)` for more information.

#### Loading
You can load the saved Cubes with 

```python
import packingcubes

cubes = packingcubes.Cubes("path/to/saved_cubes.hdf5")
```

### Searching
Currently, `packingcubes` provides multiple public methods for searching your
dataset:

```python
indices_dict = cubes.get_indices_in_sphere(particle_types, center, radius)
# particle_types is a string or Sequence[str] that maps to a particle type in
#   the snapshot
# center is anything that can be converted by numpy's array method to an (3,)
#   array, and is the sphere's center
# radius is a float
```

```python
indices_dict = cubes.get_indices_in_box(particle_types, box)
# particle_types is a string or Sequence[str] that maps to a particle type in
#   the snapshot
# box is anything that can be converted by numpy's array method to an (6,)
#   float array where the first 3 elements are the front-left-bottom corner,
#   and the second 3 elements are the box width, depth, and height 
#   (aka [x, y, z, dx, dy, dz])
```

For both methods, the returned object is a dictionary of `particle_type` keys
mapped to lists of (start, stop) tuples, representing the start and stop
indices of contiguous tuples in the **sorted** dataset. 

So the search pipeline might go something like:

```python
# cubes are associated with HDF5Dataset dataset created from orig_dataset 
# which is an h5py File object
indices_dict = cubes.get_indices_in_sphere(
    center, radius, particle_types=["PartType0", "Black_Holes"]
)

velocities_list = []
with h5py.File(orig_dataset_file) as orig_dataset:
    for start,stop in indices_dict["PartType0"]:
        shuffle = dataset.index[start:stop]
        # WARNING: the following could be very slow if shuffle gets
        # large (>10000)
        v = orig_dataset["PartType0/Velocity"][shuffle,:]
        velocities_list.extend(v)

velocities = np.fromiter(velocities_list)
```

**Warning**: this could become slow if the `shuffle` sections get big 
(`len(shuffle)>1000`)! This is because HDF5 loading is inefficient in this
manner (the `v =` line), not because of the search (the `indices_dict =`
and `shuffle =` lines). 

The sorting is already performed for the positions information (it was
necessary to construct the tree), but `packingcubes` does not apply the sort to
the other fields in the snapshot. 

So if you only need positional information, 
```python
indices_dict = cubes.get_indices_in_sphere(center, radius)

positions_list = []
# We don't need to open the orig_dataset file
dataset.particle_type = "PartType0"
for start,stop in indices_dict["PartType0"]:
    positions_list.extend(dataset.positions[start:stop])

positions = np.fromiter(positions_list)
```
should be very fast.

If you need a different field, (like the `velocities`, as above), we recommend
preloading the entire velocities array, using the shuffle list to presort it,
and then treating it analogously to the `dataset.positions` field or saving it
back out: 
```python
indices_dict = cubes.get_indices_in_sphere(center, radius)
indices = indices_dict["PartType0"]

with h5py.File(orig_dataset_path) as orig_dataset:
    loaded_velocities = orig_dataset["PartType0/velocity"]
# Just to make sure we're looking at the correct particle type
dataset.particle_type = "PartType0"
loaded_velocities = loaded_velocities[dataset.index,:]

# use directly
velocities_list = []
for start,stop in indices:
    velocities_list.extend(loaded_velocities[start:stop])

# or save it back out
with h5py.File(sorted_velocity_path, "a") as outfile:
    outfile["PartType0/velocity"] = loaded_velocities
...
velocities_list = []
with h5py.File(sorted_velocity_path) as vel_dataset:
    for start, stop in indices:
        velocities_list.extend(
            vel_dataset["PartType0/velocity"][start:stop, :]
        )


velocities = np.fromiter(velocities_list)
```

If this seems like a hassle, consider the [GUSTEAU](https://www.github.com/astrosocket/gusteau)
project, which among many other improvements, will do all of this additional
field sorting (and the original cubing) for you!


### KDTree
We have also reimplemented some of the API from the `KDTree` in `scipy.spatial`,
notably the `query_ball_point` method (This is also what the benchmarks compare
against). Example usage modified from the `scipy.spatial.KDTree` [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html):
```python
>>> import numpy as np
>>> from packingcubes import KDTree # this is the only change you need to make
>>> x, y = np.mgrid[0:5, 0:5]
>>> points = np.c_[x.ravel(), y.ravel()]
>>> tree = KDTree(points)
>>> sorted(tree.query_ball_point([2, 0], 1))
[5, 10, 11, 15]
```

#### Caveats:
* The provided dataset may be sorted in-place. See the `KDTree` constructor's
  `data` and `copy_data` arguments for more information on when that occurs. 
* Only `query`, `query_ball_point`, and `query_ball_tree` are fully supported.
  `query_pairs` and `count_neighbors` are a work in progress (`query_pairs` may
  not provide completely correct output, `count_neighbors` does not yet have
  the `KDTree` API portion implemented and may not be correct).
  `sparse_distance_matrix` is not planned. Both `count_neighbors` and
  `sparse_distance_matrix` will raise `NotImplementedErrors`.
* Only `query_ball_point` has been benchmarked and is guaranteed to be
  performant. We are currently working on `query`'s performance, but will not
  plan on beating `scipy`.
* A number of optional constructor and method arguments are not supported (for
  example, setting the distance metric `p` to a number other than 2). Some of
  these may be supported in the future (like `p=3`), others will not (like
  `balanced_tree`). For the most part, we emit warnings if an argument or its
  value does not make sense in the `packingcubes` context and try to only raise
  errors if there is no possible analog and/or a significant change in behavior.
* Only 1, 2, and 3D data is supported. PackedTrees specifically expect 3D data
  (since it's an octree implementation), thus 1 and 2D data will be copied and
  padded with 0s (e.g. `[[1, 2], [3, 4]]` will become `[[1, 2, 0], [3, 4, 0]]`
  ). This should not signficantly impact memory usage (beyond the copy) and the
  PackedTree should just function as a binary or quadtree.
* For performance reasons, some of the `packingcube` default output formats may
  not match the `scipy` output formats (`array`s instead of `list`s, for
  example). For those methods where that's a possibility, additional arguments
  can be provided which will enforce the the "proper" format (e.g. by setting
  `return_lists=True`) at a small performance penalty.
* Likewise for performance reasons, some of the the default `packingcube`
  output indices will be in terms of the sorted dataset (remember, the dataset
  will be sorted in-place unless specified otherwise). For those methods, you
  may be able to specify `return_data_inds=False` to get indices into the
  unsorted dataset. Alternatively, reference the `sort_index` property.

## Development requirements
### uv
To get ready for development, create a virtual enviroment and install the
package:
```
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

We use ruff for formatting. When you go to commit your code, it will
automatically be formatted thanks to the pre-commit hook.

Tests are performed using pytest.
### pixi
Using with pixi is pretty easy, simply
```sh
pixi shell
```
To look at visualizations, run tests, or develop, simply specify the
corresponding environment
```sh
# visualizations
pixi shell -e viz
# testing (also includes viz)
pixi shell -e test
# developing (also includes viz & test)
pixi shell -e dev
pre-commit install
```
and e.g. to run tests, say
```sh
pixi run test
```
which runs `pytest  --cov=packingcubes` in the `dev` environment.
