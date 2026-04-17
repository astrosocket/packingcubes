# ruff: noqa: D103
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import packingcubes

from ._json_parsing import as_unyt

LOGGER = logging.getLogger(__name__)

units = {
    "n": "particles",
    "creation": "s",
    "search": "us",
    "size": "bytes",
}

markers = ["s", "o", "+", "x", "*", "1", "d"]
line_styles = ["-", "--", "-.", ":", (5, 3)]

metrics = {
    "creation": {
        "packed": "tab:blue",
        "kdtree": "tab:orange",
        "cubes": "tab:purple",
        "scipy": "tab:green",
    },
    "search": {
        "brute-search": "k",
        "packed-search": "tab:blue",
        "packli-search": "tab:brown",
        "packnumb-search": "tab:olive",
        "kdtree-search": "tab:orange",
        "cubes-search": "tab:purple",
        "scipy-search": "tab:green",
    },
    # "size": {
    #     "dataset": "k",
    #     "python": "tab:blue",
    #     "packed": "tab:orange",
    #     "scipy": "tab:green",
    # },
}

particle_threshold = 400

expected = {
    "creation": lambda n: n * np.log10(n),
    "search": lambda n: np.log2(n),
    # "search": lambda n: np.sqrt(n),
    # "size":lambda n: 1.2*n/10**(np.floor(np.log10(n/particle_threshold)))
    # "size": lambda n: n,
}
expected_label = {
    "creation": r"$n\; \log(n)$",
    "search": r"$\log(n)$",
    # "size": r"$n$",
}

m_expected = {
    "search": lambda m: 1,
}
m_expected_label = {
    "search": r""  # r"$m \times$"
}

ignore_metrics = ["search:brute"]


def save_fig(
    fig: mpl.Figure,
    fig_type: str,
    *,
    save_dir: str | None = None,
    save_prefix: str | None = None,
    **kwargs,
):
    fig.suptitle(fig_type.title())
    if save_dir is None:
        return
    save_prefix = "timing" if save_prefix is None else save_prefix
    filename = Path(save_dir) / (save_prefix + "-" + fig_type + ".png")
    fig.savefig(str(filename), bbox_inches="tight")


type TSims = dict[str, dict[str, NDArray | dict]]
"""
Dictionary containing dictionaries containing timing results, organized by simulation
"""
type TPlots = tuple[list[mpl.Figure], list[mpl.Axes]]
""" Tuple of figures and axes"""


def plot_raw_times(sims: TSims, **kwargs) -> TPlots:
    """Plot raw timing results in appropriate units"""
    figs = []
    axs = []

    for i, t in enumerate(metrics):
        fig, ax = plt.subplots()
        figs.append(fig)
        axs.append(ax)
        for m, c in metrics[t].items():
            for sim in sims:
                n = sim["n"]
                if m not in sim[t]:
                    continue
                y = sim[t][m].to(units[t])
                axs[i].loglog(n, y, color=c, marker=sim["marker"], ls=sim["ls"])
            axs[i].plot(np.nan, np.nan, color=c, label=m)

        axs[i].set_xlabel(f"n [{units['n']}]")
        axs[i].set_ylabel(f"{t} [{units[t]}]")

    for sim in sims:
        axs[0].plot(
            np.nan,
            np.nan,
            color="k",
            marker=sim["marker"],
            ls=sim["ls"],
            label=sim["name"],
        )
    for ax in axs:
        ax.legend()

    for t, fig in zip(metrics, figs, strict=True):
        save_fig(fig, f"raw-{t}", **kwargs)
    return figs, axs


def plot_expected_times(sims: TSims, **kwargs) -> TPlots:
    """Plot timings in O(...) form"""
    figs = []
    axs = []

    n_scale_index = 2
    m_scale_index = 0

    for i, t in enumerate(metrics):
        fig, ax = plt.subplots()
        figs.append(fig)
        axs.append(ax)
        used_m = False
        for m, c in metrics[t].items():
            for sim in sims:
                if f"{t}:{m}" in ignore_metrics:
                    continue
                n = sim["n"]
                scale_index = min(len(n) - 1, n_scale_index)
                y = sim[t].get(m, np.full_like(n, np.nan))
                msim = sim["m"]
                if len(y.shape) > 1:
                    expected_y = np.empty_like(y)
                    for j in range(y.shape[1]):
                        expected_y[:, j] = (
                            (y[:, j] / y[scale_index, j])
                            * expected[t](n[scale_index])
                            / expected[t](n)
                            * m_expected[t](msim[m_scale_index])
                            / m_expected[t](msim[j])
                        )
                    used_m = True
                else:
                    expected_y = (
                        (y / y[scale_index])
                        * expected[t](n[scale_index])
                        / expected[t](n)
                    )
                axs[i].loglog(
                    n, expected_y, color=c, marker=sim["marker"], ls=sim["ls"]
                )
            axs[i].plot(np.nan, np.nan, color=c, label=m)

        axs[i].set_xlabel(f"n [{units['n']}]")
        axs[i].set_ylabel(f"{t}/{t}" r"$_0$ $/$ expected")
        label = (m_expected_label[t] if used_m else "") + expected_label[t]
        axs[i].text(
            0.5,
            0.1,
            label,
            transform=axs[i].transAxes,
            bbox={
                "boxstyle": "round",
                "fc": "w",
            },
        )

    for ax in axs:
        for sim in sims:
            ax.plot(
                np.nan,
                np.nan,
                color="k",
                marker=sim["marker"],
                ls=sim["ls"],
                label=sim["name"],
            )
        ax.legend()

    for t, fig in zip(metrics, figs, strict=True):
        save_fig(fig, f"expected-{t}", **kwargs)
    return figs, axs


def plot_normalized_times(sims: TSims, **kwargs) -> TPlots:
    """Plot results normalized to scipy version"""
    figs = []
    axs = []

    scale_index = 0

    for i, t in enumerate(metrics):
        fig, ax = plt.subplots()
        figs.append(fig)
        axs.append(ax)
        for m, c in metrics[t].items():
            scipy_name = "scipy" + ("-search" if "search" in m else "")
            for sim in sims:
                if f"{t}:{m}" in ignore_metrics or scipy_name not in sim[t]:
                    continue
                n = sim["n"]
                yk = sim[t][scipy_name]
                y = sim[t].get(m, np.full_like(yk, np.nan))
                msim = sim["m"]
                if len(y.shape) > 1 and len(yk.shape) == 1:
                    norm_y = np.empty_like(y)
                    for j in range(y.shape[1]):
                        norm_y[:, j] = y[:, j] / yk
                else:
                    norm_y = y / yk
                axs[i].loglog(n, norm_y, color=c, marker=sim["marker"], ls=sim["ls"])
            axs[i].plot(np.nan, np.nan, color=c, label=m)

        axs[i].set_xlabel(f"n [{units['n']}]")
        axs[i].set_ylabel(f"{t}/{t}" r"$_{\text{scipy}}$")

    for ax in axs:
        for sim in sims:
            ax.plot(
                np.nan,
                np.nan,
                color="k",
                marker=sim["marker"],
                ls=sim["ls"],
                label=sim["name"],
            )
        ax.legend()

    for t, fig in zip(metrics, figs, strict=True):
        save_fig(fig, f"normalized-{t}", **kwargs)
    return figs, axs


def plot_parallel_scaling(sims: TSims, **kwargs) -> TPlots:
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.set_figheight(10)
    labels = {}
    base_lw = 0.5
    colors = {
        "IllustrisTNG": "tab:blue",
        "LB_L10_CDM": "tab:orange",
        "ThesanXL": "tab:green",
    }
    simnames = {}
    simsizes = {}
    simmarkers = {}
    simlss = {}
    test = "kdtree"
    for sim in sims:
        if "num_threads" not in sim:
            continue
        num_threads = sim["num_threads"]
        if test not in sim["threads"]:
            continue
        core_scaling = sim["threads"][test]
        name = sim["name"]
        c = colors.get(name, "tab:gray")
        n = max(sim["n"])
        simsize = int(np.log10(n))
        label = f"$10^{{{simsize}}}$"
        if simsize not in simsizes:
            axs[1].plot(np.nan, np.nan, color="k", label=label, lw=base_lw)
            simsizes[simsize] = base_lw
            base_lw += 1
        lw = simsizes[simsize]
        simnames[name] = c
        simmarkers[name] = sim["marker"]
        simlss[name] = sim["ls"]
        axs[0].loglog(
            num_threads,
            core_scaling,
            color=c,
            marker=sim["marker"],
            ls=sim["ls"],
            lw=lw,
        )
        efficiency = core_scaling[0] / num_threads / core_scaling
        axs[1].semilogx(
            num_threads, efficiency, color=c, marker=sim["marker"], ls=sim["ls"], lw=lw
        )
    for name, c in simnames.items():
        axs[0].plot(
            np.nan,
            np.nan,
            color=c,
            ls=simlss[name],
            marker=simmarkers[name],
            label=name,
        )

    if not simnames:
        return None, None

    axs[0].xaxis.set_major_locator(
        mpl.ticker.LogLocator(
            base=2,
        )
    )
    axs[0].xaxis.set_major_formatter(mpl.ticker.LogFormatter(base=2))
    axs[0].set_ylabel("Strong Scaling [ms]")
    axs[1].set_ylabel("Parallel Efficiency ($t_1/n/t_n$)")
    axs[1].set_xlabel("Number of Cores")

    axs[0].legend()
    axs[1].legend()

    save_fig(fig, "parallel", **kwargs)
    return fig, axs


def parse_arguments(argv=None) -> dict:
    if argv is None:
        # need to skip caller or it's picked up as the snapshot file
        argv = sys.argv[1:]

    description = """
    Plot timing results from timing.py
    """
    parser = argparse.ArgumentParser(
        description=description,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase output verbosity"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"packingcubes: {packingcubes.__version__}",
    )
    plot_group = parser.add_argument_group(
        "Plot commands",
        """
        Individual arguments to specify what plots to make. 
        Specifying none is equivalent to specifying all
        """,
    )
    plot_list = ["raw", "expected", "normalized", "parallel"]
    for plot in plot_list:
        plot_group.add_argument(
            f"--plot-{plot}",
            help=f"Make a {plot} timing plot",
            dest="plot_list",
            action="append_const",
            const=plot,
        )
    parser.add_argument(
        "--name-map",
        type=str,
        help="""
        Nicer names for the timing output files. E.g. instead of 
        /blahblah/SIMNAME/extra_folder/snapshot_005.hdf5, just use SIMNAME.
        Note, must have same number of entries as the number of timing outputs.
        """,
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        help="""
        Directory to save generated figures to.
        """,
    )
    parser.add_argument(
        "-p",
        "--fig-prefix",
        type=str,
        help="""
        Prefix name for generated figures. Figures will be saved as 
        'PREFIX-raw-creation.png', 'PREFIX-parallel.png', etc.
        """,
        default="timing",
    )
    parser.add_argument(
        "timing_output",
        type=str,
        help="""
        One or more files containing timing output in JSON format. See the output of
        timing.py --save
        """,
        nargs="+",
    )
    args = parser.parse_args(argv)

    if not args.plot_list:
        args.plot_list = plot_list

    if args.name_map and len(args.name_map) != len(args.timing_output):
        raise argparse.ArgumentError(
            f"""
            Mismatch between number of name maps ({len(args.name_map)}) and
            number of timing outputs ({len(args.timing_output)})!
            """
        )

    if args.verbose >= 2:
        loglvl = logging.DEBUG
    elif args.verbose >= 1:
        loglvl = logging.INFO
    else:
        loglvl = LOGGER.level
    LOGGER.setLevel(loglvl)

    return args


def load_sim_results(
    output_list: list[str], *, name_map: list[str] | None = None
) -> dict:
    """
    Load list of output files into dictionary
    """
    sims = []
    for i, outfilepath in enumerate(output_list):
        with open(outfilepath) as outfile:
            sim = json.load(
                outfile,
                object_hook=as_unyt,
            )
        for name in ["decimations", "m", "num_threads"]:
            sim[name] = np.array(sim[name])
        snapshot_info = sim["snapshot_info"]
        sim["n"] = snapshot_info["n"] / sim["decimations"]
        sim["query"] = {}
        if "kdq-search" in sim:
            sim["query"]["kdtree"] = sim["kdq-search"]["search"]
        if "sciq-search" in sim:
            sim["query"]["scipy"] = sim["sciq-search"]["search"]
        if not sim["query"]:
            del sim["query"]
        name = name_map[i] if name_map else snapshot_info["name"]
        sim["name"] = name
        sim["marker"] = markers[i]
        sim["ls"] = line_styles[i]
        sims.append(sim)

    return sims


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig()
    LOGGER.info(f"Running with packingcubes v{packingcubes.__version__}")

    sims = load_sim_results(args.timing_output, name_map=args.name_map)
    # print(sims)

    if "raw" in args.plot_list:
        plot_raw_times(sims, save_dir=args.save_dir, save_prefix=args.fig_prefix)

    if "expected" in args.plot_list:
        plot_expected_times(sims, save_dir=args.save_dir, save_prefix=args.fig_prefix)

    if "normalized" in args.plot_list:
        plot_normalized_times(sims, save_dir=args.save_dir, save_prefix=args.fig_prefix)

    if "parallel" in args.plot_list:
        plot_parallel_scaling(sims, save_dir=args.save_dir, save_prefix=args.fig_prefix)

    if args.save_dir is None:
        plt.show()
