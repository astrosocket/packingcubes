# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: packingcubes_dev
#     language: python
#     name: packingcubes_dev
# ---

# %% [markdown]
# # Profiling with profila

# %% [markdown]
# This notebook will be used for profiling the packingcubes code using the
# [profila](www.github.com/pythonspeed/profila) tool. It will generally follow
# the approach from the Basic_Usage notebook.

# %% [markdown]
# Note that this notebook is synced with the Profiling.py file using
# [jupytext](https://github.com/mwouts/jupytext?tab=readme-ov-file#notebooks-under-version-control),
# which is checked with [Ruff](https://docs.astral.sh/ruff/), as such there may
# be places where ruff is ignored, since the ruff rules don't apply correctly to
# both cases.

# %% [markdown]
# # Initial imports and setup

# %%
# %load_ext profila

from time import time  # noqa: E402

import numpy as np  # noqa: E402

from packingcubes import Cubes, HDF5Dataset, KDTree, Optree  # noqa: E402
from packingcubes.bounding_box import make_bounding_sphere  # noqa: E402
from packingcubes.configuration import get_test_data_dir_path  # noqa: E402

# %% [markdown]
# #  Load a dataset

# %% [markdown]
# ## Specify path

# %%
simname = "IllustrisTNG"
data_path = get_test_data_dir_path()
ill_path = data_path / simname
snapfile = ill_path / "snapshot_090.hdf5"

# %% [markdown]
# ## Load the data

# %%
ds = HDF5Dataset(name=simname, filepath=snapfile)

# %% [markdown]
# ## Precompile numba components

# %%
tree = Optree(dataset=ds)
center = np.array([0, 0, 0])
radius = 1
tree.get_particle_indices_in_sphere(center=center, radius=radius)

# %%
cubes = Cubes(
    dataset=ds,
    particle_types=["PartType0"],
)

# %%
cubes.get_particle_indices_in_sphere(
    particle_types="PartType0", center=center, radius=radius
)

# %%
kdtree = KDTree(
    data=ds.positions,
)

# %%
kdtree.query_ball_point(x=center, r=radius)

# %% [markdown]
# # Run profiling

# %% [markdown]
# Here's where we'll run the actual profiling tool

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Profile tree creation

# %%
# %%profila

start = time()
# Run for 3 seconds
while (time() - start) < 3:
    tree = Optree(dataset=ds)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Profile tree search

# %%
tree = Optree(dataset=ds)
center = np.array([15000, 15000, 15000])
radius = 1000

# %%
# %%profila

start = time()
while (time() - start) < 30:
    sph_inds = tree.get_particle_indices_in_sphere(center=center, radius=radius)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Profile cubes creation

# %%
# %%profila

cubes = Cubes(
    dataset=ds,
    particle_types="PartType0",
)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Profile cubes search

# %%
cubes = Cubes(
    dataset=ds,
    particle_types="PartType0",
)
center = np.array([15000, 15000, 15000])
radius = 1000

# %%
# %%profila

start = time()
while (time() - start) < 30:
    sph_inds = cubes.get_particle_indices_in_sphere(center=center, radius=radius)

# %% [markdown]
# ## Profile KDTree search

# %%
kdtree = KDTree(
    ds.positions,
)
centers = np.array([[4753.7249348, 16280.23373589, 7140.8746265]])
radius = 856.34
ntree = kdtree._tree._tree
data = ds.data_container

# %%
sph = make_bounding_sphere(center=center, radius=radius)
sph_box = sph.bounding_box

# %%
# %%profila

start = time()
while (time() - start) < 30:
    sph_inds = ntree._get_particle_index_list_in_shape(data, sph_box, sph)

# %% [markdown]
# # Python profiling

# %%
# %load_ext pyinstrument

# %% [markdown]
# ##

# %%
kdtree = KDTree(
    ds.positions,
)
centers = np.array([[4753.7249348, 16280.23373589, 7140.8746265]])
radius = 856.34
ntree = kdtree._tree._tree
data = ds.data_container

# %%
sph = make_bounding_sphere(center=center, radius=radius)
sph_box = sph.bounding_box

# %%
# %%pyinstrument

start = time()
while (time() - start) < 30:
    sph_inds = kdtree.query_ball_point(x=centers[0], r=radius)

# %%

# %%
