"""
packingcubes visualization demo

Demo for visualizing the packingcubes cubes structure and octrees
with randomly generated points and searching
"""

import argparse
import logging
import sys
import textwrap
from functools import partial
from time import perf_counter_ns
from typing import cast

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

try:
    import pygfx as gfx  # type: ignore
    from rendercanvas import UpdateMode  # type: ignore
    from rendercanvas.auto import loop  # type: ignore
except ImportError as ie:
    print(  # noqa: T201
        "\
        This program requires the 'viz' dependency group. \
        Please check that it is installed.\
        "
    )
    raise ie

import packingcubes as pc
import packingcubes.bounding_box as bbox
import packingcubes.tree_vis as tree_vis

LOGGER = logging.getLogger("Tdemo")


help_description = """
packingcubes demo. 

Use WASD to move, QE to roll, and Space/Shift to climb/descend.
Scroll to change speed.
Use RF to move search point closer/farther.
T/G/B to toggle visibility of search point/data/tree. 
Use H to show this again. 
Ctrl+Scroll to set search size and double-click to search. 
Double-right-click will do a strict search. 
Right-click will clear search results. 
P resets everything.
Escape closes the demo.
"""

kB = 1_380_649 / 16_021_766_340_000_000_000

last_time = -1
ongoing_effects = {
    "search_position": 0,
}
search_size = 5.0


def _generate_data(
    *,
    num_particles: int | None = None,
    temperature: float | None = None,
    box: bbox.BoundingBox | None = None,
) -> tuple[NDArray, NDArray, bbox.BoundingBox]:
    """Generate particle positions and velocities in a box

    Parameters
    ----------
    num_particles: int, optional
        The number of particles to generate. Default 1000

    temperature: positive float, optional
        Mean temperature of the generated particles (in Kelvin). Particles are
        assumed to be Maxwellian, and have mass 1 GeV. Default is 1e4.

    box: BoundingBox, optional
        The bounding box of the positions. Defaults to [0,0,0,100,100,100]

    Returns
    -------
    xyz: NDArray
        The particle position data

    vxyz: NDArray
        The velocity data

    box: BoundingBox
        The bounding box of the particles

    """
    num_particles = 10000 if num_particles is None else num_particles

    temperature = 1e4 if temperature is None else temperature

    if box is None:
        box = bbox.make_bounding_box([0, 0, 0, 100, 100, 100])

    LOGGER.info(
        f"Generating {num_particles} particles within "
        f"x:{box.x}-{box.x + box.dx}, "
        f"y:{box.y}-{box.y + box.dy}, "
        f"z:{box.z}-{box.z + box.dz} "
    )
    rng = np.random.default_rng()

    # we want to split particles up into 3 gaussian blobs + background junk (5%)
    # 1 large(45%), 1 medium(30%), and 1 small(15%)
    num_large = int(0.45 * num_particles)
    num_medium = int(0.30 * num_particles)
    num_small = int(0.15 * num_particles)
    num_junk = num_particles - (num_large + num_medium + num_small)
    xyz = np.empty(shape=(num_particles, 3), dtype=float)
    vxyz = np.empty(shape=(num_particles, 3), dtype=float)

    box_center = box.midplane()
    offset = 0
    for i, num in enumerate([num_large, num_medium, num_small]):
        blob_center = np.array(
            box.project_point_on_box(
                rng.normal(loc=box_center, scale=box.size / (9 - 3 * i), size=(3,))
            )
        )
        # want roughly spheroids
        blob_radius = min(box.size) / (3 + i) ** 3
        blob_size = rng.normal(loc=blob_radius, scale=blob_radius / 4, size=(3,))
        blob = rng.normal(loc=blob_center, scale=blob_size, size=(num, 3))
        extents = np.vstack((np.min(blob, axis=0), np.max(blob, axis=0))).transpose()
        xyz[offset : (offset + num), :] = blob
        blob_temp = 10 ** (np.log10(temperature) - i)
        # assume particles have unit mass (1 GeV)
        vxyz[offset : (offset + num), :] = rng.normal(
            scale=np.sqrt(kB * blob_temp), size=(num, 3)
        )
        with np.printoptions(precision=3):
            LOGGER.info(
                f"""
Generating {num} particle blob at {blob_center} with
radius {blob_radius:.4g} and size {blob_size}.
Extents: {extents}
Temperature of {blob_temp}.
"""
            )
        offset += num

    xyz[offset:, :] = rng.uniform(size=(num_junk, 3)) * box.size + box.position

    # assume particles have unit mass (1 GeV)
    vxyz[offset:, :] = rng.normal(
        scale=np.sqrt(kB * 10 * temperature), size=(num_junk, 3)
    )

    return xyz, vxyz, box


def _cubify_data(
    xyz: NDArray, vxyz: NDArray, box: bbox.BoundingBox, *, with_cubes: bool = False
) -> tuple[pc.InMemory, pc.ParticleCubes | None]:
    """Process data through packingcubes"""
    extras = {f"v{ax}": vxyz[:, i] for i, ax in enumerate("xyz")}
    extras["velocity"] = vxyz
    if not with_cubes:
        dataset = pc.InMemory(positions=xyz, bounding_box=box)
        dataset.process_extra_fields(extra=extras)
        return dataset, None

    leafsize = 10 if len(xyz) < 40_000 else pc.octree._DEFAULT_PARTICLE_THRESHOLD
    LOGGER.info(f"Creating dataset and Cubing ({leafsize=})")
    start = perf_counter_ns()
    cubes = pc.Cubes(xyz, bounding_box=box, extras=extras, particle_threshold=leafsize)
    stop = perf_counter_ns()
    LOGGER.info(f"Done, took {(stop - start) / 1e6:.3} ms")
    return cast(pc.InMemory, cubes.dataset), cubes


def _precompile(cubes: pc.ParticleCubes):
    """Do generic search to ensure Numba portions are warmed up"""
    LOGGER.info("Doing some warmup stretches...")
    cubes.Sphere([-10, -10, -10], 1, fields="all")
    cubes.Sphere([-10, -10, -10], 1, fields="all", strict=True)
    cubes._get_packednodes_in_shape(
        pc.bounding_box.make_bounding_sphere(1, center=[-10, -10, -10])
    )


def _do_search(center: NDArray, radius: float, objects, *, strict: bool):
    cubes = objects["cubes"]
    if cubes is None:
        (xyz, vxyz, box) = objects["data"]
        dataset, cubes = _cubify_data(xyz, vxyz, box, with_cubes=True)
        objects["dataset"] = dataset
        objects["cubes"] = cubes

    scene = objects["scene"]

    for found in ["found", "found_partial", "found_entire"]:
        if found in objects:
            scene.remove(objects[found])
            del objects[found]
    LOGGER.info(f"command: Sphere({center}, {radius}, strict={strict}, fields='all')")
    start = perf_counter_ns()
    # we can't use Sphere directly here, since we want to reuse the
    # BoundingSphere
    sph = pc.bounding_box.make_bounding_sphere(center=center, radius=radius)
    sphere = cubes._Shape(shape=sph, fields="all", strict=strict)
    stop = perf_counter_ns()
    LOGGER.info(
        f"Finished Search in {(stop - start) / 1e6:.3} ms, "
        f"{len(sphere)} particles contained."
    )

    entire, partial = cubes._get_packednodes_in_shape(sph)
    LOGGER.debug(f"{len(partial)=} {len(entire)=}")

    if entire:
        tree_vis.plot_octree_mesh(entire, canvas_scene=(None, scene), cmap="Greens")
        objects["found_entire"] = scene.children[-1]
    if partial:
        tree_vis.plot_octree_mesh(partial, canvas_scene=(None, scene), cmap="Purples")
        objects["found_partial"] = scene.children[-1]

    if not len(sphere):
        return
    temp = (sphere.vx**2 + sphere.vy**2 + sphere.vz**2) / kB
    LOGGER.info(f"Min temp: {min(temp):.4g}, max: {max(temp):4g}")
    colors = mpl.colormaps.get_cmap("twilight")(
        mpl.colors.LogNorm(vmax=1e6, vmin=1, clip=True)(temp)
    )

    tree_vis.plot_positions_mesh(
        ds=sphere, canvas_scene=(None, scene), sizes=1, colors=colors
    )
    found_data = scene.children[-1]
    found_data.material.alpha_mode = "blend"
    objects["found"] = found_data


def _setup_scene(xyz: NDArray, box: bbox.BoundingBox):
    """Set up canvas/scene and add basics"""
    LOGGER.info("Setting up scene")
    canvas, scene = tree_vis.plot_box_mesh(box, color="white")
    canvas.title = help_description
    canvas.set_update_mode(UpdateMode.continuous, max_fps=60)

    objects = {}
    objects["canvas"] = canvas
    objects["scene"] = scene

    sun = gfx.PointLight()
    sun.cast_shadow = True
    sun.visible = True
    sun.local.position = box.midplane()
    sun.local.z *= 10
    # sun.look_at()
    sunhelp = gfx.PointLightHelper()
    sun.add(sunhelp)
    scene.add(sun)

    search_mat = gfx.MeshBasicMaterial(
        wireframe=True, wireframe_thickness=5, side="front", color="green"
    )
    search_mat.receive_shadow = True
    sphere_geom = gfx.tetrahedron_geometry(subdivisions=3)
    search = gfx.Mesh(sphere_geom, search_mat)
    search.visible = True
    search.local.z = -50
    search.local.euler_x = 0.1
    search.local.scale = 10

    result_mat = gfx.MeshBasicMaterial(wireframe=True, side="front", color="orange")
    result_mat.receive_shadow = True
    result = gfx.Mesh(sphere_geom, result_mat)
    result.visible = False
    scene.add(result)

    help_board = gfx.Mesh(
        gfx.box_geometry(40, 40, 1), gfx.MeshPhongMaterial(color=(0.2, 0.2, 0.2))
    )
    help_board.material.render_queue = 4000
    help_board.material.depth_test = False
    help_board.visible = True
    help_board.local.z = -40
    help_board.local.x = -5

    help_wrap = "\n".join(
        textwrap.fill(s, width=40) for s in help_description.splitlines()
    )
    help_text = gfx.Text(
        text=help_wrap,
        font_size=1.8,
        material=gfx.TextMaterial(
            color="#ddd", aa=True, depth_test=False, render_queue=4000
        ),
    )
    help_text.local.position = (0, 0, 0.5)
    help_board.add(help_text)
    help_board.alive = perf_counter_ns() + 5e9
    help_board.search_visible = search.visible
    search.visible = False

    objects["search"] = search
    objects["result"] = result
    objects["help"] = help_board

    return (canvas, scene), objects


def _plot_all_positions(canvas_scene, dataset: pc.data_objects.Dataset):
    LOGGER.info("Plotting particles")

    temp = (dataset.vx**2 + dataset.vy**2 + dataset.vz**2) / kB  # type: ignore
    LOGGER.info(f"Min temp: {min(temp):.4g}, max: {max(temp):4g}")
    colors = mpl.colormaps.get_cmap("Blues_r")(
        mpl.colors.LogNorm(vmax=1e6, vmin=10, clip=True)(temp)
    )

    tree_vis.plot_positions_mesh(
        canvas_scene=canvas_scene,
        positions=dataset.positions,
        sizes=1,
        colors=colors,
    )

    all_data = canvas_scene[1].children[-1]
    all_data.material.alpha_mode = "blend"
    all_data.material.receive_shadow = True
    return all_data


def _turn_on_help(objects):
    help_board = objects["help"]
    search = objects["search"]
    help_board.visible = True
    help_board.search_visible = search
    search.visible = False
    help_board.alive = perf_counter_ns() + 5e9


def _turn_off_help(objects):
    help_board = objects["help"]
    search = objects["search"]
    help_board.visible = False
    help_board.alive = 0
    search.visible = help_board.search_visible


def _clear_search_results(objects):
    objects["result"].visible = False
    for found in ["found", "found_partial", "found_entire"]:
        if found in objects:
            objects["scene"].remove(objects[found])
            del objects[found]


def _reset(objects):
    objects["search"].visible = True
    search_offset = -50
    ongoing_effects["search_position"] = 0
    search_size = 10
    objects["all_data"].visible = True
    _clear_search_results(objects)


def _toggle_visibility(key, objects):
    match key:
        case "t":
            objects["search"].visible ^= True
        case "g":
            objects["all_data"].visible ^= True
        case "b":
            for found in ["found_partial", "found_entire"]:
                if found in objects:
                    objects[found].visible ^= True
        case _:
            pass


def _before_draw(objects, camera):
    global last_time, ongoing_effects, search_size
    if last_time < 0:
        last_time = perf_counter_ns()
    dt = perf_counter_ns() - last_time
    search = objects["search"]
    if ongoing_effects["search_position"]:
        # update 1 per second
        search.local.z += -ongoing_effects["search_position"] * dt / 1e9
        ongoing_effects["search_position"] *= 1.05
    search.local.scale = search_size

    help_board = objects["help"]
    dt = perf_counter_ns() - help_board.alive
    if help_board.alive and dt > 0:
        _turn_off_help(objects)

    # add camera "ballast" to help stabilize after mouse movement?
    # state = camera.get_state()
    # state has position/rotation/reference_up
    # position/rotation are copies of camera.local.position/rotation

    last_time = perf_counter_ns()


def _process_key_press(event, objects):
    global ongoing_effects
    # LOGGER.info(event)
    match event["key"]:
        case "r":
            ongoing_effects["search_position"] = 1
        case "f":
            ongoing_effects["search_position"] = -1
        case "t" | "g" | "b":
            _toggle_visibility(event["key"], objects)
        case "p":
            _reset(objects)
        case "h":
            _turn_on_help(objects)
        case "Escape":
            objects["canvas"].close()
        case _:
            pass
    # LOGGER.debug(f"{search_offset=}")
    pass


def _process_key_up(event, objects):
    global ongoing_effects
    match event["key"]:
        case "r" | "f":
            ongoing_effects["search_position"] = 0
        case _:
            pass
    pass


def _process_pointer_down(event, objects):
    if event["button"] != 2:
        return
    _clear_search_results(objects)


def _process_double_click(event, objects):
    # do search
    search = objects["search"]
    result = objects["result"]
    result.visible = True
    result.world.position = search.world.position.copy()
    result.world.scale = search.world.scale

    scene = objects["scene"]

    center = result.world.position
    radius = result.world.scale[0]
    strict = event["button"] == 2
    _do_search(center, radius, objects, strict=strict)
    pass


def _process_scroll(event):
    global search_size
    if "Control" not in event["modifiers"]:
        return
    amt = -np.sign(event["dy"] or event["dx"])
    search_size *= 1.05**amt
    search_size = max(search_size, 1)
    LOGGER.debug(f"{search_size=}")
    pass


def _process_event(event, objects, gfx_stuff):
    event_type = event["event_type"]
    if event_type == "before_draw":
        _before_draw(objects, gfx_stuff["camera"])
        return
    match event_type:
        case "key_down" | "pointer_down" | "double_click" | "wheel":
            if objects["help"].alive:
                _turn_off_help(objects)
        case _:
            pass
    match event_type:
        case "key_down":
            _process_key_press(event, objects)
        case "key_up":
            _process_key_up(event, objects)
        case "pointer_down":
            _process_pointer_down(event, objects)
        case "double_click":
            _process_double_click(event, objects)
        case "wheel":
            _process_scroll(event)
        case _:
            return
    gfx_stuff["renderer"].request_draw()
    LOGGER.debug(event)


def main(argv=None):
    """Run demo"""
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    args = parse_args(argv)

    xyz, vxyz, box = _generate_data(
        num_particles=args.num_particles, temperature=args.temperature, box=args.box
    )

    # no cubes for faster startup?
    dataset, cubes = _cubify_data(xyz, vxyz, box, with_cubes=not args.fast_startup)
    if not args.fast_startup:
        _precompile(cubes)

    canvas_scene, objects = _setup_scene(xyz, box)

    all_data = _plot_all_positions(canvas_scene, dataset)

    objects["all_data"] = all_data
    objects["data"] = (xyz, vxyz, box)
    objects["dataset"] = dataset
    objects["cubes"] = cubes

    LOGGER.info("Rendering scene")
    canvas, gfx_stuff = tree_vis.display_scene(canvas_scene)

    gfx_stuff["camera"].show_object(gfx_stuff["scene"].children[-1])
    camera = gfx_stuff["camera"]
    camera.add(objects["search"])
    camera.add(objects["help"])
    canvas_scene[1].add(camera)
    canvas.add_event_handler(
        partial(_process_event, objects=objects, gfx_stuff=gfx_stuff), "*"
    )
    LOGGER.info("Ready!")
    loop.run()

    return


def parse_args(argv=None):
    """Parse CLI arguments"""
    if argv is None:
        # skip caller
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=help_description)
    parser.add_argument(
        "-n",
        "--num_particles",
        help="Number of particles to use",
        type=int,
    )
    parser.add_argument(
        "-T",
        "--temperature",
        help="Temperature of particles",
        type=float,
    )
    parser.add_argument(
        "--fast-startup",
        help=(
            "Skip initial cubing and precompilation. Note that this will cause "
            "the demo to hang when you first perform the various searches, but "
            "the demo should recover."
        ),
        action="store_true",
    )
    box_args = parser.add_argument_group(
        "Box parameters",
        """
        Arguments to change the dimensions of bounding box of the particles
        """,
    )
    for ax in "xyz":
        box_args.add_argument(f"-{ax}", help=f"minimum {ax} coordinate", type=float)
        box_args.add_argument(
            f"-{ax.upper()}", help=f"size in the {ax} direction", type=float
        )
    args = parser.parse_args(argv)
    box = [0, 0, 0, 100, 100, 100]
    for i, ax in enumerate("xyz"):
        x = getattr(args, ax, None)
        if x is not None:
            box[i] = x
        dx = getattr(args, ax.upper(), 100)
        if dx is not None:
            box[3 + i] = dx
    try:
        box = bbox.make_bounding_box(box)
    except bbox.BoundingBoxError as bbe:
        parser.error(bbe)
    args.box = box
    LOGGER.info(help_description)
    LOGGER.warning(
        "First activation of cubing/searching will take longer due to JIT compile\n"
    )
    LOGGER.info("Additional command line arguments are available, try `--help`!")
    return args


if __name__ == "__main__":
    sys.exit(main())
