# packingcubes

[![PyPI](https://img.shields.io/pypi/v/packingcubes)](https://pypi.org/project/packingcubes)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/packingcubes)](https://pypi.org/project/packingcubes/)

<!--- Tests and style --->
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)


Compact octree implementation for fast search.

`packingcubes` aims to provide a fast, minimal-memory-usage octree
implementation, specialized for use in astronomical/astrophysical contexts.
It's written in pure python, with [Numba](https://numba.pydata.org/)-based
acceleration of the critical code paths.

View the documentation at [packingcubes.readthedocs.io](https://packingcubes.readthedocs.io)!

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

#### Optional packages
Visualization (the `viz` group):

* `matplotlib` - to use basic octree visualization and plotting (see
  [Finding Particles Within a Shape](https://packingcubes.readthedocs.io/en/latest/Users-Guide/Tutorials/ParticlesWithinShape/)
  or Basic_Usage in the Examples)
* `pygfx` - to do interactive octree visualization (see Example_PackedTree in
   the Examples)
* `rendercanvas` - to do interactive octree visualization
* `pyside6` - performant interactive octree visualization (uses Qt, other
   options include `wgpu-py`, `pyodide`. See the
  [rendercanvas backends](https://rendercanvas.readthedocs.io/stable/backends.html)
   documentation for more details

Jupyter (the `jupyter` group):

* `jupyter-rfb` - for interactive octree visualization in a notebook (see the Visualization section, above)

The `all` group combines both of the above.

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
<!-- --8<-- [start:basic-const-snap-disk] -->
From the command line:

```sh
packcubes SNAPSHOT OUTPUT
```
will generate a `Cubes` data structure and store it in the `OUTPUT` hdf5 file
along with the sorted positions and shuffle-list.

Alternatively:
```python
import packingcubes

cubes = packingcubes.Cubes("path/to/snapshot.hdf5", save_dataset=True)

cubes.save("path/to/output.hdf5")

# sorted positions/indices are stored in snapshot_sorted.hdf5 by default
# to save elsewhere, use
cubes = packingcubes.Cubes(
    "path/to/snapshot.hdf5", 
    save_dataset=True,
    sorted_filepath="path/to/output.hdf5"
)
```
<!-- --8<-- [end:basic-const-snap-disk] -->

#### From positions in memory
If you already have `positions_data` as an Nx3 matrix, you can use

``` python
cubes = packingcubes.Cubes(positions_data)
```

Note: this data will be sorted in place! You may want to make a copy first.

You can also do the following for easy saving. 
The data will still not be copied (by default)

``` python
cubes = packingcubes.Cubes(
    positions_data,
    sorted_filepath="path/to/dataset_output.hdf5",
    # filepath would also work
    save_dataset=True
)
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
dataset. Some examples:

```python
indices = cubes.get_indices_in_sphere(center, radius)
# center is anything that can be converted by numpy's array method to an (3,)
#   array, and is the sphere's center
# radius is a float
```

or

```python
indices = cubes.get_index_list_in_box(box)
# box is anything that can be converted by numpy's array method to an (6,)
#   float array where the first 3 elements are the front-left-bottom corner,
#   and the second 3 elements are the box width, depth, and height 
#   (aka [x, y, z, dx, dy, dz])
```

or 

```python
sphere = cubes.Sphere(center, radius, fields="all")
# center and radius have the same meaning as above
```

For the first method, the returned object is an array with 3 columns. Each row can
be considered a chunk of data, which looks like `[start, stop, partial]` 
representing the start and stop indices of contiguous data in the 
**sorted** dataset. The third column, `partial` denotes whether the chunk
was partially (`1`) or entirely (`0`) contained within the sphere/box.

In the second method, the returned object is a list of the actual particle indices.

In the third, the returned object is a subdataset of all the fields present that
are attached to the particles in the defined sphere.

So the search pipeline might go something like:

```python
dataset = packingcubes.GadgetishHDF5Dataset(orig_dataset_file)
cubes = packingcubes.Cubes(
    dataset,
    particle_type="PartType0",
    extras={"v":"Velocity"}
)

sphere = cubes.Sphere(center, radius, fields="all")

velocities = sphere.v
```

or as a "one-liner"[^1]:
```python
velocities = packingcubes.Cubes(
    orig_dataset_file,
    particle_type="PartType0",
    extras={"v":"Velocity"}
).Sphere(center, radius, fields="all").v
```

[^1]: Okay, it's _technically_ 5 lines. But only because it's broken up for clarity!

If this seems like a hassle, consider the [GUSTEAU](https://www.github.com/astrosocket/gusteau)
project, which among many other improvements, will do all of this additional
field sorting (and the original cubing) for you!


### KDTree
<!-- --8<-- [start:KDTree] -->
We have also reimplemented some of the API from the `KDTree` in `scipy.spatial`,
notably the `query_ball_point` method (This is also what the benchmarks compare
against). Example usage modified from the `scipy.spatial.KDTree` [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html):
```python
>>> import numpy as np
>>> from packingcubes import OpTree as KDTree # this is the only change you need to make
>>> x, y = np.mgrid[0:5, 0:5]
>>> points = np.c_[x.ravel(), y.ravel()]
>>> tree = KDTree(points)
>>> sorted(tree.query_ball_point([2, 0], 1))
[5, 10, 11, 15]
```

<!-- --8<-- [end:KDTree]  -->

#### Caveats:
* The provided dataset may be sorted in-place. See the `OpTree` constructor's
  `data` and `copy_data` arguments for more information on when that occurs. 
* Only `query` and `query_ball_point` are fully supported. `query_ball_tree`,
  `query_pairs` and `count_neighbors` are a work in progress and will raise
  `NotImplementedErrors` until fully implemented. `sparse_distance_matrix` is
  not planned and will always raise `NotImplementedErrors`.
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
