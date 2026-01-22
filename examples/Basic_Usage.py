# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: packingcubes-jupyter
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Basic PackedTree usage

# %% [markdown]
# In this notebook we'll demonstrate the basic usage of the PackedTree library.

# %% [markdown]
# Note that this notebook is synced with the Basic_Usage.py file using
# [jupytext](https://github.com/mwouts/jupytext?tab=readme-ov-file#notebooks-under-version-control),
# which is checked with [Ruff](https://docs.astral.sh/ruff/), as such there are
# places where ruff is ignored, since the ruff rules don't apply correctly to
# both cases.

# %% [markdown]
# # Initial Imports and setup

# %%
# ruff: noqa: T201
import matplotlib.pyplot as plt
import numpy as np

from packingcubes import HDF5Dataset, Optree
from packingcubes.configuration import get_test_data_dir_path

# %% [markdown]
# # Load a Dataset

# %% [markdown]
# ## Specify dataset path

# %% [markdown]
# We will load data from the test data directory.
# This can be specified using e.g. the minimal `packingcubes.toml` file:
# ```toml
# # could be located at ./packingcubes.toml
# test_data_dir = "~/packing_cubes_test_data"
# ```

# %%
simname = "IllustrisTNG"
data_path = get_test_data_dir_path()
ill_path = data_path / simname
snapfile = ill_path / "snapshot_090.hdf5"

# %% [markdown]
# ## Load dataset

# %% [markdown]
# We'll only use the first 10% of the data so everything runs faster

# %%
ds = HDF5Dataset(name=simname, filepath=snapfile)
# decimate the data so it loads/runs faster
ds._positions = ds._positions[: int(len(ds) / 10), :]
ds._setup_index()

# %% [markdown]
# # Create tree

# %%
tree = Optree(dataset=ds)


# %%

# %% [markdown]
# # Show node filling fraction

# %% [markdown]
# Define the filling fraction as
# $\frac{\text{\# of particles in node}}{\text{particle threshold}}$.
#
# Then for fun, we can plot the filling fraction per node for all of the leaves
# in the tree


# %%
def node_filling_amount(tree, *, normalized=True):
    bins = np.zeros((tree.particle_threshold,))
    leaves = [*tree.get_leaves()]
    for leaf in tree.get_leaves():
        leaf_size = leaf.node_end - leaf.node_start
        bins[leaf_size] = bins[leaf_size] + 1
    if normalized:
        bins = bins / len(leaves)
    return bins


normalized = True
bins = node_filling_amount(tree, normalized=normalized)

plt.figure(figsize=(10, 4))
plt.bar(
    np.linspace(1, tree.particle_threshold, num=tree.particle_threshold)
    / tree.particle_threshold
    * 100,
    bins,
    width=100 / tree.particle_threshold,
)
plt.xlabel("Filling Fraction (%)")
if normalized:
    plt.ylabel("Fraction of Nodes")
else:
    plt.ylabel("Number of Nodes")

# %%
