import logging
from collections.abc import Iterable
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pygfx as gfx  # type: ignore
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
from numpy.typing import NDArray
from rendercanvas.auto import RenderCanvas  # type: ignore

import packingcubes.bounding_box as bbox
import packingcubes.data_objects as data_objects
import packingcubes.octree as octree

LOGGER = logging.getLogger(__name__)


"""
Module for visualizing octrees and particle data

"""


def _extreme_nodes(nodes: Iterable[octree.OctreeNode]):
    """
    Find deepest/shallowest nodes (longest/shortest tag) in list of OctreeNodes
    """
    shallowest = deepest = None
    for node in nodes:
        if deepest is None or len(node.tag) > len(deepest.tag):
            deepest = node
        if shallowest is None or len(node.tag) < len(shallowest.tag):
            shallowest = node
    return deepest, shallowest


def _get_faces(box: bbox.BoundingBox) -> NDArray:
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
    box_vertices = box.get_box_vertices()

    inds = [
        [0, 1, 5, 4, 0],  # front
        [2, 0, 4, 6, 2],  # left
        [2, 6, 7, 3, 2],  # back
        [1, 3, 7, 5, 1],  # right
        [4, 5, 7, 6, 4],  # up
        [0, 2, 3, 1, 0],  # down
    ]
    return np.array([[box_vertices[j] for j in i] for i in inds])


def _get_geometry(box: bbox.BoundingBox) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Return the 24 vertices, 12 index sets, and 24 normals corresponding to this box
    """
    vertices = (
        np.unpackbits(
            np.array([154, 242, 194, 125, 106, 96, 53, 248, 50], dtype=np.uint8)
        )
        .reshape((24, 3))
        .astype("float32")
    )
    vertices *= box.box[3:]
    vertices += box.box[:3]
    indices = np.array(
        [
            [0, 1, 2],
            [3, 2, 1],
            [4, 5, 6],
            [7, 6, 5],
            [8, 9, 10],
            [11, 10, 9],
            [12, 13, 14],
            [15, 14, 13],
            [16, 17, 18],
            [19, 18, 17],
            [20, 21, 22],
            [23, 22, 21],
        ],
        dtype=np.uint32,
    )
    normals = np.zeros((24, 3), dtype=np.float32)
    normals[:4, 0] = normals[8:12, 1] = normals[16:20, 2] = 1
    normals[4:8, 0] = normals[12:16, 1] = normals[20:, 2] = -1
    texcoords = (
        np.unpackbits(np.array([114, 114, 114, 114, 114, 114], dtype=np.uint8))
        .reshape((24, 2))
        .astype("float32")
    )
    return vertices, indices, normals, texcoords


def _get_quad_geometry(
    box: bbox.BoundingBox,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Return the 8 vertices, 6 index sets, and 6 normals corresponding to this box
    """
    vertices = box.get_box_vertices(1).astype(np.float32)

    indices = np.array(
        [
            [
                0,
                1,
                5,
                4,
            ],  # front
            [
                2,
                0,
                4,
                6,
            ],  # left
            [
                2,
                6,
                7,
                3,
            ],  # back
            [
                1,
                3,
                7,
                5,
            ],  # right
            [
                4,
                5,
                7,
                6,
            ],  # up
            [
                0,
                2,
                3,
                1,
            ],  # down
        ],
        dtype=np.uint32,
    )
    normals = np.array(
        [
            [0, -1, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.float32,
    )
    texcoords = np.zeros((6, 1), dtype=np.float32)
    return vertices, indices, normals, texcoords


def cubify_tree_Poly3D(
    tree: octree.Octree | list[octree.OctreeNode],
    *,
    leaves_only: bool = True,
    cmap: mpl.colors.Colormap | None = None,
) -> dict[int, Poly3DCollection]:
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

    vertex_dict: dict[int, list[Any]] = {}

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


def plot_box_poly(
    box: bbox.BoundingBox,
    *,
    ax: Axes3D | None = None,
    color: ColorType | None = None,
):
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


def plot_octreenode_poly(
    node: octree.OctreeNode,
    *,
    ax: Axes3D | None = None,
    color: ColorType | None = None,
):
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
    return plot_box_poly(node.box, ax=ax, color=color)


def plot_octree_poly(
    tree: octree.Octree | list[octree.OctreeNode],
    *,
    ax: Axes3D | None = None,
    cmap: mpl.colors.Colormap | None = None,
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

    poly_dict = cubify_tree_Poly3D(tree, cmap=cmap, leaves_only=leaves_only)

    for poly in poly_dict.values():
        ax.add_collection3d(poly)

    ax.set_aspect("equal")

    return ax


def plot_positions_poly(
    *,
    positions: NDArray | None = None,
    ds: data_objects.Dataset | None = None,
    ax: Axes3D | None = None,
):
    """
    Plot an 3D scatter plot of the positions in a dataset

    Creates a new figure if ax is not provided.

    Args:
        positions: NDArray, optional
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


def cubify_tree_geom(
    tree: octree.Octree | list[octree.OctreeNode],
    *,
    leaves_only: bool = True,
    cmap: mpl.colors.Colormap | None = None,
) -> dict[int, gfx.Group]:
    """
    Transform an octree into a dict of Meshes indexed by node depth

    Each Mesh shares a single material, so we must return them
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
        mesh_dict: dict[int, Mesh]
        Dictionary of Meshes
    """
    if cmap is None:
        cmap = mpl.colormaps["plasma"]
    elif isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if hasattr(tree, "get_leaves") and leaves_only:
        nodes = list(tree.get_leaves())
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

    # use relative depth for colors
    color_norm = mpl.colors.Normalize(vmin=min_depth, vmax=max_depth)
    # use absolute norm for the alpha channel
    alpha_norm = mpl.colors.Normalize(vmin=0, vmax=max_depth)

    num_nodes = len(nodes)
    positions = np.zeros((num_nodes, 3), dtype=np.float32)
    size_dict: dict[int, NDArray] = {}
    color_dict: dict[int, tuple[float, float, float, float]] = {}
    depths = np.zeros((num_nodes,), dtype=int)

    for i, node in enumerate(nodes):
        positions[i, :] = node.box.box[:3]

        depth = len(node.tag)
        depths[i] = depth
        if depth not in size_dict:
            size_dict[depth] = node.box.box[3:]
            # alpha values between 0.1 and 0.5
            color_dict[depth] = cmap(
                color_norm(depth), alpha=alpha_norm(depth) * 0.4 + 0.1
            )

    group_dict = {}
    for depth, size in size_dict.items():
        box_geom = gfx.box_geometry()
        default_material = gfx.MeshBasicMaterial(
            color=color_dict[depth], wireframe=True, wireframe_thickness=5
        )
        group_dict[depth] = gfx.Group()
        for pos in positions[depths == depth, :]:
            cube = gfx.Mesh(box_geom, default_material)
            cube.local.position = pos + size / 2
            cube.local.scale = size
            group_dict[depth].add(cube)

    return group_dict


def cubify_tree_mesh(
    tree: octree.Octree | list[octree.OctreeNode],
    *,
    leaves_only: bool = True,
    cmap: mpl.colors.Colormap | None = None,
) -> dict[int, gfx.Mesh]:
    """
    Transform an octree into a dict of Meshes indexed by node depth

    Each Mesh shares a single material, so we must return them
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
        mesh_dict: dict[int, Mesh]
        Dictionary of Meshes
    """

    if cmap is None:
        cmap = mpl.colormaps["plasma"]
    elif isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if hasattr(tree, "get_leaves") and leaves_only:
        nodes = list(tree.get_leaves())
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

    geometry_dict: dict[int, list[Any]] = {}

    # use relative depth for colors
    color_norm = mpl.colors.Normalize(vmin=min_depth, vmax=max_depth)
    # use absolute norm for the alpha channel
    alpha_norm = mpl.colors.Normalize(vmin=0, vmax=max_depth)

    for node in nodes:
        vertices, indices, normals, texcoords = _get_quad_geometry(node.box)

        depth = len(node.tag)
        # alpha values between 0.1 and 0.5
        color = cmap(color_norm(depth), alpha=alpha_norm(depth) * 0.4 + 0.1)

        if depth in geometry_dict:
            geometry_dict[depth][1] += 1
            geometry_dict[depth][2] = np.vstack((geometry_dict[depth][2], vertices))
            geometry_dict[depth][3] = np.vstack(
                (
                    geometry_dict[depth][3],
                    indices + len(vertices) * geometry_dict[depth][1],
                )
            )
            geometry_dict[depth][4] = np.vstack((geometry_dict[depth][4], normals))
            geometry_dict[depth][5] = np.vstack((geometry_dict[depth][5], texcoords))
        else:
            geometry_dict[depth] = [color, 0, vertices, indices, normals, texcoords]

    mesh_dict = {}
    for k, v in geometry_dict.items():
        color, _, vertices, indices, normals, texcoords = v

        min_vert, max_vert = np.min(vertices, axis=0), np.max(vertices, axis=0)
        size = max_vert - min_vert

        # Normalize and center vertices
        normalized_vertices = (vertices - min_vert) / size - 0.5

        geometry = gfx.Geometry(
            positions=normalized_vertices,
            indices=indices,
            normals=normals,
            texcoords=texcoords,
        )
        material = gfx.MeshBasicMaterial(
            color=color,
            wireframe=True,
            wireframe_thickness=3,
        )
        mesh_dict[k] = gfx.Mesh(geometry, material)
        mesh_dict[k].local.position = min_vert + size / 2
        mesh_dict[k].local.scale = size

    return mesh_dict


def setup_scene() -> tuple[RenderCanvas, gfx.Scene]:
    canvas = RenderCanvas(present_method="screen")

    # Setup basic scene
    scene = gfx.Scene()

    scene.add(gfx.Background.from_color("#000"))

    # add grid
    grid = gfx.Grid(
        None,
        gfx.GridMaterial(
            major_step=100,
            minor_step=10,
            thickness_space="world",
            major_thickness=2,
            minor_thickness=0.1,
            infinite=True,
        ),
        orientation="xz",
    )
    grid.local.y = -120
    scene.add(grid)

    # add ambient light source
    ambient = gfx.AmbientLight()
    scene.add(ambient)

    return canvas, scene


def plot_box_mesh(
    box: bbox.BoundingBox,
    *,
    canvas_scene: tuple[RenderCanvas, gfx.Scene] | None = None,
    color: ColorType | None = None,
    show_normals: bool | ColorType = True,
):
    """
    Plot a single BoundingBox

    Creates a new canvas and scene if not provided

    Args:
        box: bounding_box.BoundingBox
        The box to plot

        canvas_scene: tuple[Canvas, Scene], optional
        The Canvas and Scene to plot on. Default None

        color: ColorType, optional
        The color of the cube. Can be any valid matplotlib color (str, array,
        etc.) Default "green".

        show_normals: bool | ColorType, optional
        Whether to show cube normals. If True (default), normals are shown in
        red. Can instead specify a color to display normals in that color.

    Returns:
        canvas_scene: tuple[Canvas, Scene]
        The canvas-scene tuple
    """
    if canvas_scene is None:
        canvas_scene = setup_scene()

    if color is None:
        color = "green"

    if not isinstance(show_normals, bool):
        normal_color = mpl.colors.to_rgba(show_normals)
    elif show_normals:
        normal_color = mpl.colors.to_rgba("red")

    canvas, scene = canvas_scene

    _color = mpl.colors.to_rgba(color)

    unit_box = bbox.make_bounding_box([-0.5, -0.5, -0.5, 1, 1, 1])

    vertices, indices, normals, texcoords = _get_quad_geometry(unit_box)

    geometry = gfx.Geometry(
        positions=vertices, indices=indices, normals=normals, texcoords=texcoords
    )

    material = gfx.MeshBasicMaterial(
        color=_color, wireframe=True, wireframe_thickness=3
    )

    cube = gfx.Mesh(geometry, material)
    cube.local.position = box.box[:3] + box.box[3:] / 2
    cube.local.scale = box.box[3:]

    scene.add(cube)

    if show_normals:
        normal_material = gfx.MeshNormalLinesMaterial(
            color=normal_color, line_length=0.5
        )

        normal_cube = gfx.Mesh(geometry, normal_material)
        normal_cube.local.position = box.box[:3] + box.box[3:] / 2
        normal_cube.local.scale = box.box[3:]

        scene.add(normal_cube)

    return canvas_scene


def plot_octreenode_mesh(
    node: octree.OctreeNode,
    *,
    canvas_scene: tuple[RenderCanvas, gfx.Scene] | None = None,
    color: ColorType | None = None,
) -> tuple[RenderCanvas, gfx.Scene]:
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
    return plot_box_mesh(
        node.box, canvas_scene=canvas_scene, color=color, show_normals=False
    )


def plot_octree_mesh(
    tree: octree.Octree | list[octree.OctreeNode],
    *,
    canvas_scene: tuple[RenderCanvas, gfx.Scene] | None = None,
    cmap: mpl.colors.Colormap | None = None,
    leaves_only: bool = True,
) -> tuple[RenderCanvas, gfx.Scene]:
    """
    Plot an Octree or other list of OctreeNodes

    Creates a new canvas if canvas_scene is not provided.

    Nodes are colored according to their depth, so all nodes at the same depth
    will have the same color.

    Args:
        tree: octree.Octree | List[OctreeNode]
        The tree or list of OctreeNodes to plot

        canvas_scene: tuple[RenderCanvas, Scene], optional
        The 3D canvas to plot on. Default None

        cmap:
        The colormap to use for plotting. Can be any valid matplotlib colormap
        (str, array, etc.) Default "plasma".

        leaves_only: bool, optional
        For Octrees, whether to plot only the leaves (Default) or the entire
        tree

    Returns:
        ax: Axes3D
    """
    if canvas_scene is None:
        canvas_scene = setup_scene()

    canvas, scene = canvas_scene

    mesh_dict = cubify_tree_mesh(tree, cmap=cmap, leaves_only=leaves_only)

    group = gfx.Group()
    for mesh in mesh_dict.values():
        group.add(mesh)

    scene.add(group)

    return canvas_scene


def plot_positions_mesh(
    *,
    positions: NDArray | None = None,
    ds: data_objects.Dataset | None = None,
    canvas_scene: tuple[RenderCanvas, gfx.Scene] | None = None,
    sizes: NDArray | int | None = None,
    colors: NDArray | None = None,
) -> tuple[RenderCanvas, gfx.Scene]:
    """
    Plot an 3D scatter plot of the positions in a dataset

    Creates a new figure if ax is not provided.

    Args:
        positions: NDArray, optional
        Array of 3D points to plot. If not provided will use ds.positions.
        Default None

        ds: data_objects.Dataset, optional
        Dataset from which to pull positions from. **Must** be provided if
        positions is None. Default None

        canvas_scene: tuple[RenderCanvas, Scene], optional
        The 3D canvas to plot on. Default None

        sizes: int, NDArray, optional
        Size of particles to plot. Can specify an array to have individual
        sizes. Default 3

        colors: NDArray, optional
        Array of per-particle colors. Defaults to xkcd:skyblue

    Returns:
        ax: Axes3D
    """
    if canvas_scene is None:
        canvas_scene = setup_scene()
    canvas, scene = canvas_scene

    if positions is None:
        if ds is None:
            raise ValueError("Must provide ds if positions is not provided!")
        positions = ds.positions

    # size 3, xkcd:sky blue color, alpha=1
    if sizes is None:
        sizes = 3 * np.ones((len(positions), 1), dtype=np.float32)
    elif isinstance(sizes, int):
        sizes = sizes * np.ones((len(positions), 1), dtype=np.float32)

    if colors is None:
        colors = np.ones_like(positions.astype(np.float32))
        colors[:, 0] = 0.4588235294117647
        colors[:, 1] = 0.7333333333333333
        colors[:, 2] = 0.9921568627450981

    geometry = gfx.Geometry(
        positions=positions.astype(np.float32),
        sizes=sizes.astype(np.float32),
        colors=colors.astype(np.float32),
    )

    material = gfx.PointsGaussianBlobMaterial(
        color_mode="vertex",
        size_mode="vertex",
        size_space="world",
    )

    points = gfx.Points(geometry, material)

    scene.add(points)

    return canvas_scene


def display_scene(canvas_scene: tuple[RenderCanvas, gfx.Scene]) -> RenderCanvas:
    canvas, scene = canvas_scene
    renderer = gfx.renderers.WgpuRenderer(canvas)
    camera = gfx.PerspectiveCamera(70)
    camera.show_object(scene)
    controller = gfx.FlyController(camera, register_events=renderer)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    return (
        canvas,
        {
            "scene": scene,
            "renderer": renderer,
            "camera": camera,
            "controller": controller,
        },
    )
