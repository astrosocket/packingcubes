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
# # Benchmarking Plots

# %%

# %% [markdown]
# # Imports

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%

# %% [markdown]
# # Data
# (Note: master data source for Illustris and LB_L10_CDM in notes. This may not
# be up to date)

# %% [markdown]
# ## IllustrisTNG

# %%
n = [1.5e5, 1.5e6, 1.5e7]

creation = {
    "data resetting": [
        1.13e-3,
        0.012,
        0.127,
    ],
    "python": [
        0.0482,
        0.54,
        6.49,
    ],
    "packed": [
        0.012,
        0.184,
        2.17,
    ],
    "cubes": [
        0.016,
        0.147,
        2.00,
    ],
    "kdtree": [
        0.0208,
        0.300,
        4.33,
    ],
}

search = {
    "python": [
        1.25,
        4.49,
        17.8,
    ],
    "packed": [
        0.090,
        0.250,
        0.884,
    ],
    "pack_list": [0.162, 2.58, 21.9],
    "cubes": [
        0.0587,
        0.104,
        0.224,
    ],
    "kdtree": [
        0.108,
        1.47,
        15.9,
    ],
}

size = {
    "dataset": [
        1871217,
        18710649,
        187104885,
    ],
    "python": [
        545461,
        5231664,
        50810413,
    ],
    "packed": [
        38088,
        359177,
        3468337,
    ],
    "kdtree": [
        5063559,
        50484807,
        508383527,
    ],
}

illustris = {
    "name": "IllustrisTNG",
    "creation": creation,
    "search": search,
    "size": size,
    "n": n,
    "marker": "s",
    "ls": "-",
}

# %% [markdown]
# ## Simba

# %%
n = [1.5e5, 1.5e6, 1.5e7]

creation = {
    "data resetting": [
        1.15e-3,
        12.3e-3,
        134e-3,
    ],
    "python": [
        0.0482,
        0.575,
        6.65,
    ],
    "packed": [
        0.0379,
        0.460,
        5.53,
    ],
    "cubes": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "kdtree": [
        0.0186,
        0.231,
        2.82,
    ],
}

search = {
    "python": [
        1.26,
        4.87,
        20.3,
    ],
    "packed": [
        0.87,
        2.07,
        4.66,
    ],
    "pack_list": [np.nan, np.nan, np.nan],
    "cubes": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "kdtree": [
        0.523,
        3.68,
        32.9,
    ],
}

size = {
    "dataset": [
        1909377,
        19092225,
        190920693,
    ],
    "python": [
        475785,
        4987588,
        50087794,
    ],
    "packed": [
        33288,
        342857,
        3423397,
    ],
    "kdtree": [
        5165319,
        51502343,
        518559015,
    ],
}

simba = {
    "name": "SIMBA",
    "creation": creation,
    "search": search,
    "size": size,
    "n": n,
    "marker": "x",
    "ls": "-",
}

# %% [markdown]
# ## Swift-EAGLE

# %%
n = [1.5e5, 1.5e6, 1.5e7]

creation = {
    "data resetting": [
        1.22e-3,
        14.3e-3,
        142e-3,
    ],
    "python": [
        0.050,
        0.578,
        6.59,
    ],
    "packed": [
        0.0405,
        0.463,
        5.42,
    ],
    "cubes": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "kdtree": [
        0.0186,
        0.244,
        3.05,
    ],
}

search = {
    "python": [
        3.21,
        12.2,
        52.2,
    ],
    "packed": [
        1.7,
        4.68,
        12.2,
    ],
    "pack_list": [np.nan, np.nan, np.nan],
    "cubes": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "kdtree": [
        1.06,
        8.02,
        90.2,
    ],
}

size = {
    "dataset": [
        3813237,
        38130837,
        381306861,
    ],
    "python": [
        545559,
        4974569,
        48502428,
    ],
    "packed": [
        38088,
        341617,
        3312897,
    ],
    "kdtree": [
        5158217,
        51431113,
        517846503,
    ],
}

sweagle = {
    "name": "Swift-EAGLE",
    "creation": creation,
    "search": search,
    "size": size,
    "n": n,
    "marker": "+",
    "ls": "-",
}

# %% [markdown]
# ## LB_L10_CDM

# %%
n = [
    1.1e5,
    1.1e6,
    1.1e7,
    1.1e8,
    1.1e9,
]

creation = {
    "data resetting": [2.4e-3, 0.0407, 0.325, 5.05, 25.8],
    "python": [0.180, 0.866, 11, 101, 1100],
    "packed": [0.0133, 0.167, 2.12, 17.5, 223.2],
    "cubes": [
        0.02,
        0.0998,
        1.11,
        12,
        130.8,
    ],
    "kdtree": [0.0402, 0.393, 4.79, 38, 403.2],
}

search = {
    "python": [2.5, 8.7, 61, 208, 850],
    "packed": [0.257, 0.667, 2.31, 9.44, 32],
    "pack_list": [0.328, 0.795, 3.83, 27.3, 216],
    "cubes": [0.187, 0.256, 0.572, 1.92, 8.0],
    "kdtree": [0.248, 2.94, 33.2, 463, 2340],
}

size = {
    "dataset": [2577165, 25769973, 257698221, 2576980557, 25769803917],
    "python": [174704, 2062787, 27842084, 328732478, 3499040395],
    "packed": [12488, 142417, 1908497, 22438977, 237917597],
    "kdtree": [3510121, 34949961, 348316393, 3511474721, 34963718503],
}

extra = {
    "notes": [
        (
            "Run on nvdimm(DR,PyOct creation & search) & snapshot 6/"
            "icx otherwise, snapshot 13"
        ),
        (
            "Run on nvdimm(DR,PyOct creation & search) & snapshot 6/"
            "icx otherwise, snapshot 13"
        ),
        (
            "Run on nvdimm(DR,PyOct creation & search) & snapshot 6/"
            "icx otherwise, snapshot 13"
        ),
        (
            "Run on nvdimm(DR,PyOct creation & search) & snapshot 6/"
            "icx otherwise, snapshot 13"
        ),
        (
            "Run on nvdimm(DR,PyOct creation & search) & snapshot 6/"
            "icx otherwise, snapshot 13"
        ),
    ]
}

LB_L10_CDM = {
    "name": "LB_L10_CDM",
    "creation": creation,
    "search": search,
    "size": size,
    "n": n,
    "marker": "o",
    "ls": "--",
    "extra": extra,
}

# %% [markdown]
# ## Metadata

# %%
units = {"n": "particles", "creation": "s", "search": "ms", "size": "bytes"}

# %%
metrics = {
    "creation": {
        "data resetting": "k",
        "python": "tab:blue",
        "packed": "tab:orange",
        "cubes": "tab:purple",
        "kdtree": "tab:green",
    },
    "search": {
        "python": "tab:blue",
        "packed": "tab:orange",
        "pack_list": "tab:brown",
        "cubes": "tab:purple",
        "kdtree": "tab:green",
    },
    "size": {
        "dataset": "k",
        "python": "tab:blue",
        "packed": "tab:orange",
        "kdtree": "tab:green",
    },
}


# %%
particle_threshold = 400

expected = {
    "creation": lambda n: n * np.log10(n),
    # "search": lambda n: np.log10(n),
    "search": lambda n: np.sqrt(n),
    # "size":lambda n: 1.2*n/10**(np.floor(np.log10(n/particle_threshold)))
    "size": lambda n: n,
}
expected_label = {
    "creation": r"$n\; \log(n)$",
    "search": r"$\log(n)$",
    # "search": r"$\sqrt{n}$",
    # "size":(
    #     r"$1.2\times\frac{n}{10^{\left\lfloor n/"
    #     f"{particle_threshold}"
    #     r"\right\rfloor}}$")
    "size": r"$n$",
}

ignore_metrics = ["creation:data resetting"]

# %% [markdown]
# # Plotting

# %% [markdown]
# ## Sim choice

# %%
sims = [illustris, LB_L10_CDM, simba, sweagle]
sims = [illustris, LB_L10_CDM]
# sims = [LB_L10_CDM]

# %% [markdown]
# ## Raw

# %% jupyter={"source_hidden": true}
figs = []
axs = []

for i, t in enumerate(metrics):
    fig, ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)
    for m, c in metrics[t].items():
        for sim in sims:
            n = np.array(sim["n"])
            y = np.array(sim[t][m])
            axs[i].loglog(n, y, color=c, marker=sim["marker"], ls=sim["ls"])
        axs[i].plot(np.nan, np.nan, color=c, label=m)

    axs[i].set_xlabel(f"n [{units['n']}]")
    axs[i].set_ylabel(f"{t} [{units[t]}]")

for sim in sims:
    axs[0].plot(
        np.nan, np.nan, color="k", marker=sim["marker"], ls=sim["ls"], label=sim["name"]
    )
for ax in axs:
    ax.legend()

# %% [markdown]
# ## Expected

# %% jupyter={"source_hidden": true}
figs = []
axs = []

scale_index = 2

for i, t in enumerate(metrics):
    fig, ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)
    for m, c in metrics[t].items():
        for sim in sims:
            if f"{t}:{m}" in ignore_metrics:
                continue
            y = np.array(sim[t][m])
            n = np.array(sim["n"])
            expected_y = (
                (y / y[scale_index]) * expected[t](n[scale_index]) / expected[t](n)
            )
            axs[i].loglog(n, expected_y, color=c, marker=sim["marker"], ls=sim["ls"])
        axs[i].plot(np.nan, np.nan, color=c, label=m)

    axs[i].set_xlabel(f"n [{units['n']}]")
    axs[i].set_ylabel(f"{t}/{t}" r"$_0$ $/$ expected")
    axs[i].text(
        0.5,
        0.1,
        expected_label[t],
        transform=axs[i].transAxes,
        bbox={
            "boxstyle": "round",
            "fc": "w",
        },
    )

for sim in sims:
    axs[0].plot(
        np.nan, np.nan, color="k", marker=sim["marker"], ls=sim["ls"], label=sim["name"]
    )
for ax in axs:
    ax.legend()

# %% [markdown]
# ## Normalized

# %%
figs = []
axs = []

scale_index = 0

for i, t in enumerate(metrics):
    fig, ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)
    for m, c in metrics[t].items():
        for sim in sims:
            if f"{t}:{m}" in ignore_metrics:
                continue
            y = np.array(sim[t][m])
            yk = np.array(sim[t]["kdtree"])
            n = np.array(sim["n"])
            axs[i].loglog(n, y / yk, color=c, marker=sim["marker"], ls=sim["ls"])
        axs[i].plot(np.nan, np.nan, color=c, label=m)

    axs[i].set_xlabel(f"n [{units['n']}]")
    axs[i].set_ylabel(f"{t}/{t}" r"$_{\text{kdtree}}$")

for sim in sims:
    axs[0].plot(
        np.nan, np.nan, color="k", marker=sim["marker"], ls=sim["ls"], label=sim["name"]
    )
for ax in axs:
    ax.legend()

# %%
