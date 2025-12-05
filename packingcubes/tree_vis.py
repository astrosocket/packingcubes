import logging
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree

LOGGER = logging.getLogger(__name__)

"""
Module for visualizing octrees and particle data

"""


def _extreme_nodes(nodes: List[octree.OctreeNode]):
    """
    Find deepest/shallowest nodes (longest/shortest tag) in list of OctreeNodes
    """
    deepest = nodes[0]
    shallowest = nodes[0]
    for node in nodes:
        if len(node.tag) > len(deepest.tag):
            deepest = node
        if len(node.tag) < len(shallowest.tag):
            shallowest = node
    return deepest, shallowest


def _get_faces(box: bbox.BoundingBox) -> List[np.ndarray]:
    """
    Return the 30 vertices of the 6 box faces

    Note 30 vertices because 6 faces * (4+1) vertices per face (the +1 is so
    the polygon is closed)

    Args:
        box: bounding_box.BoundingBox
        Box to get faces of

    Returns:
        vertices: List[numpy.ndarray]
        Returned vertices are in the form: list[30 x 3 ndarrays]
        Vertices are orderd such that the normal faces outwards
    """
    box_vertices = bbox.get_box_vertices(box)

    inds = [
        [0, 1, 5, 4, 0],  # front
        [2, 0, 4, 6, 2],  # left
        [2, 6, 7, 3, 2],  # back
        [1, 3, 7, 5, 1],  # right
        [4, 5, 7, 6, 4],  # up
        [0, 2, 3, 1, 0],  # down
    ]
    vertices = np.array([[box_vertices[j] for j in i] for i in inds])

    return vertices


def cubify_tree(
    tree: octree.Octree | List[octree.OctreeNode],
    *,
    leaves_only=True,
    cmap: mpl.colors.Colormap = None,
) -> Dict[int, Poly3DCollection]:
    """
    Transform an octree into a dict of Poly3DCollections indexed by node depth

    Each Poly3DCollection shares a single color, so we must return them
    separately.

    Args:
        tree: octree.Octree | list[octree.OctreeNode]
        The octree (or list of OctreeNodes) to convert

        leaves_only: boolean, optional
        Whether to only include leaves or the full tree. Default True

        cmap: str | matplotlib.colors.Colormap, optional
        The colormap to use for the rendered octree. A string will be passed to
        the built-in colormaps (default is matplotlib.colormaps['plasma'])

    Returns:
        poly_dict: dict[int, Poly3DCollection]
        Dictionary of Poly3DCollections
    """

    if cmap is None:
        cmap = mpl.colormaps["plasma"]
    elif isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if hasattr(tree, "get_leaves") and leaves_only:
        nodes = tree.get_leaves()
    else:
        nodes = list(tree)

    # find extreme nodes, this will set the color limits
    deepest, shallowest = _extreme_nodes(nodes)
    min_depth = len(shallowest.tag)
    max_depth = len(deepest.tag)
    # For monochrome color maps, using 4 or fewer colors doesn't look as nice
    # use the middle colors instead
    if max_depth - min_depth <= 4:
        max_depth += 1
        min_depth -= 1

    vertex_dict = {}

    # use relative depth for colors
    color_norm = mpl.colors.Normalize(vmin=min_depth, vmax=max_depth)
    # use absolute norm for the alpha channel
    alpha_norm = mpl.colors.Normalize(vmin=0, vmax=max_depth)

    for node in nodes:
        vertices = _get_faces(node.box)

        depth = len(node.tag)
        # alpha values between 0.1 and 0.5
        color = cmap(color_norm(depth), alpha=alpha_norm(depth) * 0.4 + 0.1)

        if depth in vertex_dict:
            vertex_dict[depth][1] = np.vstack((vertex_dict[depth][1], vertices))
        else:
            vertex_dict[depth] = [color, vertices]

    poly_dict = {}
    for k, v in vertex_dict.items():
        color, vertices = v
        poly_dict[k] = Poly3DCollection(vertices, facecolor=color, edgecolor=color)
        poly_dict[k].set_facecolor("#00000000")

    return poly_dict


def plot_box(box: bbox.BoundingBox, *, ax=None, color=None):
    """
    Plot a single BoundingBox

    Creates a new figure if ax is not provided

    Args:
        box: bounding_box.BoundingBox
        The box to plot

        ax: Axes3D, optional
        The 3D axes to plot on. Default None

        color:
        The color of the cube. Can be any valid matplotlib color (str, array,
        etc.) Default "green".

    Returns:
        ax: Axes3D
    """
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    if color is None:
        color = "green"

    vertices = _get_faces(box)

    cube = Poly3DCollection(vertices, facecolor=color, edgecolor=color)
    cube.set_facecolor("#00000000")

    ax.add_collection3d(
        cube,
    )

    ax.set_aspect("equal")

    return ax


def plot_octreenode(node: octree.OctreeNode, *, ax=None, color=None):
    """
    Plot a single OctreeNode

    Effectively just plot_box(node.box, ...)
    Creates a new figure if ax is not provided

    Args:
        node: octree.OctreeNode
        The node to plot

        ax: Axes3D, optional
        The 3D axes to plot on. Default None

        color:
        The color of the cube. Can be any valid matplotlib color (str, array,
        etc.) Default "green".

    Returns:
        ax: Axes3D
    """
    return plot_box(node.box, ax=ax, color=color)


def plot_octree(
    tree: octree.Octree | List[octree.OctreeNode],
    *,
    ax=None,
    cmap: mpl.colors.ColorMap = None,
    leaves_only: bool = True,
):
    """
    Plot an Octree or other list of OctreeNodes

    Creates a new figure if ax is not provided.

    Nodes are colored according to their depth, so all nodes at the same depth
    will have the same color.

    Args:
        tree: octree.Octree | List[OctreeNode]
        The tree or list of OctreeNodes to plot

        ax: Axes3D, optional
        The 3D axes to plot on. Default None

        cmap:
        The colormap to use for plotting. Can be any valid matplotlib colormap
        (str, array, etc.) Default "plasma".

        leaves_only: bool, optional
        For Octrees, whether to plot only the leaves (Default) or the entire
        tree

    Returns:
        ax: Axes3D
    """
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    poly_dict = cubify_tree(tree, cmap=cmap, leaves_only=leaves_only)

    for poly in poly_dict.values():
        ax.add_collection3d(poly)

    ax.set_aspect("equal")

    return ax


def plot_positions(
    *,
    positions=None,
    ds=None,
    ax=None,
):
    """
    Plot an 3D scatter plot of the positions in a dataset

    Creates a new figure if ax is not provided.

    Args:
        positions: ArrayLike, optional
        Array of 3D points to plot. If not provided will use ds.positions.
        Default None

        ds: data_objects.Dataset, optional
        Dataset from which to pull positions from. **Must** be provided if
        positions is None. Default None

        ax: Axes3D, optional
        The 3D axes to plot on. Default None

    Returns:
        ax: Axes3D
    """
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    if positions is None:
        if ds is None:
            raise ValueError("Must provide ds if positions is not provided!")
        positions = ds.positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=3).set_alpha(0.1)

    ax.set_aspect("equal")

    return ax
