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
# # PackedTree Example

# %% [markdown]
# Example PackedTree visualization using the functions in tree_vis and the
# IllustrisTNG dataset

# %% [markdown]
# Note that this notebook is synced with the Basic_Usage.py file using
# [jupytext](https://github.com/mwouts/jupytext?tab=readme-ov-file#notebooks-under-version-control),
# which is checked with [Ruff](https://docs.astral.sh/ruff/), as such there are
# places where ruff is ignored, since the ruff rules don't apply correctly to
# both cases.

# %% [markdown]
# # **WARNING**

# %% [markdown]
# To see visualizations in this notebook, the jupyter-rfb package must be
# installed (e.g. `pip install jupyter-rfb` or
# `conda install -c conda-forge jupyter-rfb` or `pixi add jupyter-rfb`)
# **in the environment running the jupyter server**. It should already be
# included in the packingcubes environments.

# %% [markdown]
# # Initial imports and setup

# %%
# ruff: noqa: T201
from functools import partial
from pathlib import Path

import h5py  # for additional data loading
import matplotlib as mpl
import numpy as np

try:
    import pygfx as gfx
    from rendercanvas.auto import loop
except ImportError as ie:
    print(
        "This script requires the 'viz' dependency group. "
        "Please ensure it is installed."
    )
    raise ie

# We import these internals for additional display capabilities
import packingcubes.bounding_box as bbox
import packingcubes.packed_tree as optree
import packingcubes.tree_vis as tree_vis
from packingcubes import HDF5Dataset, Optree

# %%
simname = "IllustrisTNG"
data_path = Path("~/socket/snaps").expanduser()
ill_path = data_path / simname
snapfile = ill_path / "snapshot_090.hdf5"

# %% [markdown]
# # Load data files and create trees

# %% [markdown]
# ## Load data

# %%
print("Loading data files")
ds = HDF5Dataset(name=simname, filepath=snapfile)
# decimate the data so it loads/runs faster
decimation_factor = 10
new_length = int(len(ds) / decimation_factor)
ds._positions = ds._positions[:new_length, :]
ds._setup_index()

# We'll also load density and temperature data, for fun
with h5py.File(ds.filepath) as file:
    masses = np.array(file[ds.particle_type]["Masses"][:new_length])
    density = np.array(file[ds.particle_type]["Density"][:new_length])
    temperature = np.array(file[ds.particle_type]["InternalEnergy"][:new_length])


# %% [markdown]
# ## Create actual particle bounding box

# %%
positions = ds.positions
min_pos, max_pos = np.min(positions, axis=0), np.max(positions, axis=0)
positions_box = bbox.make_bounding_box(np.array([*min_pos, *(max_pos - min_pos)]))

# %% [markdown]
# ## Create tree

# %%
print("Creating tree. Note this takes a few seconds to JIT compile code.")
# Uncomment to use a PythonOctree instead
# import packingcubes.octree as octree
# tree = octree.PythonOctree(
#     dataset=ds,
# )
tree = Optree(
    dataset=ds,
)

# %%
# Note that we need to manually sort temperature and density because they are
# not included normally
density = density[ds.index]
temperature = temperature[ds.index]
masses = masses[ds.index]

# %% [markdown]
# # Draw Everything

# %%
print("Begin drawing")

# %% [markdown]
# ## Generate initial scene
# We don't need to select anything here, so we can just use what is
# provided by tree_viz

# %%
print("Plotting dataset bounding box")
canvas_scene = tree_vis.plot_box_mesh(ds.bounding_box, color="white")
canvas_scene[0].title = (
    "Packing Trees simulation. "
    "Use WASD to move, QE to roll, and Space/Shift to climb/descend. "
    "Turn Particles on/off with F"
)

# %% [markdown]
# ## Add outlines

# %%
outline = gfx.BoxHelper(thickness=5, color="#fa0")
selected_node = gfx.BoxHelper(thickness=5, color="#33FF33")
hover_material = gfx.MeshPhongMaterial(color="#FFAA00", pick_write=True)


# %% [markdown]
# ## Define callbacks and helper functions


# %% jupyter={"source_hidden": true}
def to_aabb(box: bbox.BoundingBox) -> np.typing.NDArray:
    b = box.box
    aabb = np.zeros((2, 3), dtype=np.float32)
    aabb[0, :] = b[:3]
    aabb[1, :] = b[:3] + b[3:]
    return aabb


# %% jupyter={"source_hidden": true}
def set_material(material, obj):
    if isinstance(obj, gfx.Mesh) and hasattr(obj, f"{material}_material"):
        obj.material = getattr(obj, f"{material}_material")


# %% jupyter={"source_hidden": true}
selected_point = None
default_point_color = None
selected_point_color = gfx.Color("#fa0").rgba


def select_point(event):
    info = event.pick_info
    if "vertex_index" in info:
        i = round(info["vertex_index"])
        points = event.target
        if not isinstance(points, gfx.Points):
            return
        if not hasattr(points, "default_point_color"):
            points.default_point_color = points.geometry.colors.data[i, :]
        if hasattr(points, "previous_selection"):
            points.geometry.colors.data[points.previous_selection, :] = (
                points.default_point_color
            )
            points.geometry.colors.update_range(points.previous_selection)
        points.previous_selection = i
        points.geometry.colors.data[i, :] = selected_point_color
        points.geometry.colors.update_range(i)
        canvas_scene.request_draw()
    pass


# %% jupyter={"source_hidden": true}
def hover_point(event):
    # This will be very similar to select_node
    obj = event.target
    if event.type == "pointer_enter":
        # get closest vertex
        info = event.pick_info
        if "vertex_index" in info:
            i = round(info["vertex_index"])
            points = event.target

            point = points.geometry.data[i, :]
            size = points.geometry.sizes.data[i]
            aabb = np.array([point - size, point + size], dtype=np.float32)

            # set selected node
            obj.add(outline)
            outline.set_transform_by_aabb(aabb, scale=1.1)
    elif outline.parent:
        outline.parent.remove(outline)


# %% jupyter={"source_hidden": true}
selected_levels = []


def select_level(event):
    obj = event.current_target

    event.stop_propagation()

    # clear selection
    if selected_levels:
        while selected_levels:
            ob = selected_levels.pop()
            ob.traverse(partial(set_material, "default"))

    # clear node selection if present
    if selected_node.parent:
        selected_node.parent.remove(selected_node)

    # if the background, particle, or grid was clicked, we're done
    if (isinstance(obj, (gfx.Renderer, gfx.Grid, gfx.Points))) or (
        isinstance(obj, gfx.Mesh) and not hasattr(obj, "selected_material")
    ):
        return

    if isinstance(obj, gfx.Group):
        print("Selecting the group")

    # set selection (group or mesh)
    selected_levels.append(obj)
    obj.traverse(partial(set_material, "selected"))

    if isinstance(obj, gfx.Mesh) and hasattr(obj, "selected_material"):
        pos = event.target.local.position
        size = event.target.local.scale

        centroid = pos - size / 2

        node = tree._tree._get_containing_node_of_point(centroid)
        if node:
            # aabb = to_aabb(node.box)
            selected_node.scale = 1.05
            # set selected node
            print(f"Selecting node {optree.get_name(node)} at {centroid}")
            obj.add(selected_node)
            selected_node.set_transform_by_object(obj, "local", scale=1.05)
            # selected_node.set_transform_by_aabb(aabb, scale=1.1)


# %% jupyter={"source_hidden": true}
def select_node(event):
    # when this event handler is invoked on non-leaf nodes of the
    # scene graph, event.target will still point to the leaf node that
    # originally triggered the event, so we use event.current_target
    # to get a reference to the node that is currently handling
    # the event, which can be a Mesh, a Group or None (the event root)
    obj = event.current_target

    # prevent propagation so we handle this event only at one
    # level in the hierarchy
    event.stop_propagation()

    # clear level selection if present
    if selected_levels and hasattr(selected_levels[0], "default_material"):
        selected_levels[0].material = selected_levels[0].default_material
        selected_levels.pop()

    # clear node selection if present
    if selected_node.parent:
        selected_node.parent.remove(selected_node)

    # if the background, particle, or grid was clicked, we're done
    if (isinstance(obj, (gfx.Renderer, gfx.Grid, gfx.Points))) or not hasattr(
        obj, "selected_material"
    ):
        return

    # get closest node

    pos = event.target.local.position
    size = event.target.local.scale

    centroid = pos - size / 2

    node = tree._tree._get_containing_node_of_point(centroid)
    if node:
        # aabb = to_aabb(node.box)
        selected_node.scale = 1.05
        # set selected node
        obj.add(selected_node)
        # selected_node.set_transform_by_aabb(aabb, scale=1.1)


# %% jupyter={"source_hidden": true}
def hover_node(event):
    # This will be very similar to select_node
    obj = event.target
    if event.type == "pointer_enter":
        # get closest node
        obj.add(outline)
        outline.set_transform_by_object(obj, "local", scale=1.1)
        pos = event.target.local.position
        size = event.target.local.scale

        centroid = pos - size / 2

        node = tree._tree._get_containing_node_of_point(centroid)
        if node:
            print(f"Hovering over {optree.get_name(node)} at position {centroid}")
        else:
            print(f"No node found for position {centroid}")
        #     # aabb = to_aabb(node.box)

        #     if outline.parent:
        #         outline.parent.remove(outline)

        #     # set selected node
        #     obj.add(outline)
        #     outline.local.scale = 1.1
        # outline.set_transform_by_aabb(aabb, scale=1.1)
    elif outline.parent:
        # print("Un-hovering node")
        outline.parent.remove(outline)


# %%
show_particles = True


def on_key(e):
    global show_particles
    if e.key == "f":
        show_particles = not show_particles
    particles.visible = show_particles


# %% [markdown]
# ## Add particles

# %%
smoothing_lengths = np.cbrt(3 * masses / (4 * np.pi * density)) * 4
colors = mpl.colormaps.get_cmap("twilight")(
    mpl.colors.LogNorm(vmax=2429385.0 * 0.1, vmin=0.13396032 * 10, clip=True)(
        temperature
    )
)
ALPHA_NORM = 0.05
alphas = mpl.colors.LogNorm()(1.0 / smoothing_lengths**2)
alphas *= ALPHA_NORM

colors[:, 3] = alphas

# colors = None
# smoothing_lengths = None

# %%
print("Plotting positions")
tree_vis.plot_positions_mesh(
    ds=ds, canvas_scene=canvas_scene, sizes=smoothing_lengths, colors=colors
)

particles = canvas_scene[1].children[-1]

# points = canvas_scene[1].children[-1]
# points.add_event_handler(select_point,"click")
# points.add_event_handler(hover_point,"pointer_enter", "pointer_leave")
# also add particle bounding box
# this should be identical to the octree root box and the ds bounding box
# when all particles are included
if decimation_factor > 1:
    print("Plotting bounding box of actual data")
    tree_vis.plot_box_mesh(positions_box, canvas_scene=canvas_scene, color="gray")

# %% [markdown]
# ## Add Octree

# %%
print("Plotting tree.")
leaves = [*tree.get_leaves()]
if len(leaves) > 1e4:
    print(
        f"Note this may take a long time because we are rendering {len(leaves)} nodes."
    )
del leaves
# only plot leaves for now
tree_vis.plot_octree_mesh(tree, canvas_scene=canvas_scene, leaves_only=True)

# Don't use plot_octree_mesh so that we have better access to materials
# and can add callbacks
# group_dict = tree_vis.cubify_tree_geom(tree, leaves_only=True)

# canvas_scene[1].add(outline)
# canvas_scene[1].add(selected_node)
# for group in group_dict.values():
#     group.add_event_handler(select_level, "double_click")
#     for cube in group.children:
#         original_material = cube.material
#         original_material.pick_write = True
#         original_material.wireframe = True
#         selected_material = gfx.MeshBasicMaterial(
#             color="#33AA33",
#             pick_write=True,
#             wireframe=True,
#         )
#         cube.default_material = original_material
#         cube.selected_material = selected_material
#         cube.add_event_handler(select_level, "click")
#         cube.add_event_handler(hover_node, "pointer_enter", "pointer_leave")
#     canvas_scene[1].add(group)

# %% [markdown]
# ## Render scene

# %%
print("Rendering scene")
canvas, gfx_stuff = tree_vis.display_scene(canvas_scene)

gfx_stuff["camera"].show_object(gfx_stuff["scene"].children[-1])
gfx_stuff["renderer"].add_event_handler(on_key, "key_down")
canvas  # noqa

# %%

# %% [markdown]
# ## Start animation if run as script

# %%
if __name__ == "__main__":
    print("Ready!")
    loop.run()

# %%

# %%

# %%
