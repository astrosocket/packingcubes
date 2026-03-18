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

# %%
ds = HDF5Dataset(name=simname, filepath=snapfile)

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
cumulative = True
bins = node_filling_amount(tree, normalized=normalized)

plt.figure(figsize=(10, 4))
plt.bar(
    np.linspace(1, tree.particle_threshold, num=tree.particle_threshold)
    / tree.particle_threshold
    * 100,
    np.cumsum(bins) if cumulative else bins,
    width=100 / tree.particle_threshold,
)
plt.xlabel("Filling Fraction (%)")
if normalized:
    plt.ylabel("Fraction of Nodes")
else:
    plt.ylabel("Number of Nodes")


# %% [markdown]
# # Plot as image
# For fun, since the tree is just a simple array of uint32, with the last four
# field representing a length (so omittable if necessary), we can treat it as an
# image directly.


# %%
def find_best_size(array_len):
    """
    Given integer l, find the closest height and width such that width*height<l
    and height < width < 2*height. If the second condition doesn't hold, just
    do the best we can
    """
    # start with the square root, since that's the best we can do
    width = int(np.ceil(np.sqrt(array_len)))
    min_width = width / 2
    while width > min_width:
        height = width
        lg = width * height
        while lg >= array_len - 1:
            height -= 1
            lg = width * height
        if array_len - 1 <= lg <= array_len:
            break
        width -= 1
    return width, height


w, h = find_best_size(len(tree._tree.tree))
img = np.reshape(tree.packed_form[: (w * h)].copy(), (w, h))
a = 255 - ((img & 0xFF_00_00_00) >> 24)
r = (img & 0xFF_00_00) >> 16
g = (img & 0xFF_00) >> 8
b = img & 0xFF
rgba = np.dstack((r, g, b, a))
print(rgba.shape)
ratio = w / h
figscale = 8
fig = plt.figure(figsize=(figscale * ratio, figscale))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.imshow(
    rgba,
    vmin=0,
)
plt.show()

# %%

# %%
