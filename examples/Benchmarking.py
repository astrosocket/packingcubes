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
m = [100, 1000, 10000, 100000, 1_000_000]

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
        0.00275,
        0.0337,
        0.360,
    ],
    "kdtree": [
        0.005,
        0.039,
        0.343,
    ],
    "cubes": [
        0.007,
        0.038,
        0.346,
    ],
    "scipy": [
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
        0.0329,
        0.102,
        0.3811,
    ],
    "pack_list": [0.0418, 0.1849, 2.3762],
    "kdtree": [0.0417, 0.0832, 2.3758],
    "cubes": [
        0.027,
        0.0392,
        0.0911,
    ],
    "scipy": [
        0.118,
        1.66,
        20.1,
    ],
}

search_cn = {
    "brute": [
        [2.03, 2.08, 2.04, np.nan, np.nan],
        [24, 24.3, 23.6, np.nan, np.nan],
        [236, 243, 238, np.nan, np.nan],
    ],
    "packed": [
        [0.017, 0.026, 0.0651, 0.1488, 0.0716],
        [0.0205, 0.0299, 0.0794, 0.2967, 0.5611],
        [0.022, 0.0432, 0.0942, 0.2974, 1.0759],
    ],
    "pack_list": [
        [0.0203, 0.0314, 0.0869, 0.251, 0.133],
        [0.024, 0.0357, 0.1086, 0.4941, 1.398],
        [0.0253, 0.0504, 0.1367, 0.5073, 2.2656],
    ],
    "kdtree": [
        [0.0356, 0.0416, 0.0938, 0.1639, np.nan],
        [0.0417, 0.0487, 0.1076, 0.3542, 1.1493],
        [0.119, 0.0718, 0.1347, 0.52, 1.7581],
    ],
    "cubes": [
        [0.0285, 0.0375, 0.0551, 0.0903, 0.0592],
        [0.0327, 0.0403, 0.0906, 0.2284, 0.299],
        [0.0315, 0.0575, 0.1165, 0.2771, 0.731],
    ],
    "scipy": [
        [6.3418e-03, 2.1304e-02, 1.7510e-01, 1.2945e00, np.nan],
        [7.0537e-03, 2.3151e-02, 2.3133e-01, 2.0613e00, 1.6306e01],
        [6.5945e-03, 2.6757e-02, 2.3498e-01, 1.8143e00, 1.7211e01],
    ],
}

numba_only = {"packed": [0.0315, 0.0299, 0.0402]}

num_cores = [1, 2, 4, 8, 16]
core_scaling = {
    "cubes": [
        {"n": 1.6e6, "m": 1e4, "t": [0.0378, 0.0368, 0.0352, 0.0471, 0.0409]},
        {"n": 1.6e7, "m": 100, "t": [0.0125, 0.0126, 0.0125, 0.0148, 0.020]},
        {"n": 1.6e7, "m": 1e5, "t": [0.0989, 0.0972, 0.0964, 0.126, 0.237]},
    ],
    "kdtree-search": [
        {"n": 1.6e6, "m": 1e4, "t": [0.140, 0.101, 0.086, 0.114, 0.099]},
        {"n": 1.6e7, "m": 100, "t": [0.0242, 0.0248, 0.025, 0.034, 0.039]},
        {"n": 1.6e7, "m": 1e5, "t": [0.597, 0.451, 0.387, 0.442, 0.434]},
        {"n": 1.6e7, "m": 1e6, "t": [3.046, 2.140, 1.802, 1.772, 1.787]},
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
    "scipy": [
        5063559,
        50484807,
        508383527,
    ],
}

illustris = {
    "name": "IllustrisTNG",
    "creation": creation,
    "search": search,
    "search_cn": search_cn,
    "size": size,
    "numba_only": numba_only,
    "num_cores": num_cores,
    "core_scaling": core_scaling,
    "n": n,
    "m": m,
    "marker": "s",
    "ls": "-",
}

# %% [markdown]
# ## Simba

# %%
n = [1.5e5, 1.5e6, 1.5e7]
m = [100, 1000, 10000]

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
    "kdtree": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "cubes": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "scipy": [
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
    "kdtree": [np.nan, np.nan, np.nan],
    "cubes": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "scipy": [
        0.523,
        3.68,
        32.9,
    ],
}


search_cn = {
    "python": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "packed": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "pack_list": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "kdtree": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "cubes": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "scipy": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
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
    "scipy": [
        5165319,
        51502343,
        518559015,
    ],
}

simba = {
    "name": "SIMBA",
    "creation": creation,
    "search": search,
    "search_cn": search_cn,
    "size": size,
    "n": n,
    "m": m,
    "marker": "x",
    "ls": "-",
}

# %% [markdown]
# ## Swift-EAGLE

# %%
n = [1.5e5, 1.5e6, 1.5e7]
m = [100, 1000, 10000]

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
    "kdtree": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "cubes": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "scipy": [
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
    "kdtree": [np.nan, np.nan, np.nan],
    "cubes": [
        np.nan,
        np.nan,
        np.nan,
    ],
    "scipy": [
        1.06,
        8.02,
        90.2,
    ],
}


search_cn = {
    "python": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "packed": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "pack_list": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "kdtree": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "cubes": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ],
    "scipy": [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
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
    "scipy": [
        5158217,
        51431113,
        517846503,
    ],
}

sweagle = {
    "name": "Swift-EAGLE",
    "creation": creation,
    "search": search,
    "search_cn": search_cn,
    "size": size,
    "n": n,
    "m": m,
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
m = [100, 1000, 10000, 100_000, 1_000_000, 10_000_000]

creation = {
    "data resetting": [2.4e-3, 0.0407, 0.325, 5.05, 25.8],
    "python": [0.180, 0.866, 11, 101, 1100],
    "packed": [0.00422, 0.0506, 0.676, 7.66, 89.4],
    "kdtree": [
        0.00443,
        0.049,
        0.683,
        7.72,
        94.8,
    ],
    "cubes": [
        0.0174,
        0.0431,
        0.396,
        3.89,
        41.7,
    ],
    "scipy": [0.0264, 0.303, 3.77, 46.6, 561],
}

search = {
    "python": [2.5, 8.7, 61, 208, 850],
    "packed": [0.0943, 0.2334, 0.8667, 3.8573, 16.1042],
    "pack_list": [1.4520e-01, 4.9833e-01, 5.3897e00, 2.7640e01, 1.8095e02],
    "kdtree": [0.17145, 0.51316, 5.8192, 25.868, 181.12],
    "cubes": [0.114, 0.1514, 0.2442, 0.9408, 2.8417],
    "scipy": [0.257, 2.71, 36.4, 370, 3570],
}

search_cn = {
    "python": [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ],
    "packed": [
        [0.0616, 0.067, 0.0601, 0.0741, 0.0798, 0.0767],
        [0.0763, 0.0917, 0.1536, 0.0748, 0.177, 0.1723],
        [0.0859, 0.0993, 0.191, 0.5105, 0.1193, 0.5533],
        [0.0935, 0.1153, 0.208, 0.6113, 1.8325, 0.3268],
        [0.0977, 0.1105, 0.2278, 0.7722, 2.9517, 12.1397],
    ],
    "pack_list": [
        [0.0709, 0.0908, 0.0691, 0.0989, 0.1095, 0.0997],
        [0.0787, 0.1081, 0.2412, 0.2021, 0.3703, 0.3616],
        [0.0882, 0.1255, 0.2864, 1.0174, 1.0355, 1.7002],
        [0.0949, 0.1328, 0.3116, 1.1756, 4.1899, 23.5773],
        [0.1114, 0.146, 0.3643, 1.4212, 6.0595, 44.1897],
    ],
    "kdtree": [
        [0.0811, 0.1005, 0.0795, np.nan, np.nan, np.nan],
        [0.0893, 0.117, 0.2646, 0.2151, np.nan, np.nan],
        [0.0985, 0.1376, 0.3009, 1.0306, 1.0509, np.nan],
        [0.1049, 0.1435, 0.3259, 1.2153, 4.0881, 23.6259],
        [0.1223, 0.1564, 0.3802, 1.4381, 6.0231, 44.2495],
    ],
    "cubes": [
        [0.1061, 0.1242, 0.1885, 0.176, 0.1842, 0.1763],
        [0.099, 0.1105, 0.1504, 0.1775, 0.2214, 0.2238],
        [0.0962, 0.1064, 0.1037, 0.2274, 0.2264, 0.3759],
        [0.103, 0.1013, 0.1308, 0.1776, 0.5545, 0.3313],
        [0.105, 0.0885, 0.0915, 0.0946, 0.7726, 1.8603],
    ],
    "scipy": [
        [0.017, 0.068, 0.358, np.nan, np.nan, np.nan],
        [0.017, 0.073, 0.433, np.nan, np.nan, np.nan],
        [0.018, 0.072, 0.445, 3.499, 46.656, np.nan],
        [0.022, 0.078, 0.46, 3.844, 46.758, 537.854],
        [0.022, 0.088, 0.503, 3.911, 50.929, 550.015],
    ],
}

size = {
    "dataset": [2577165, 25769973, 257698221, 2576980557, 25769803917],
    "python": [174704, 2062787, 27842084, 328732478, 3499040395],
    "packed": [12488, 142417, 1908497, 22438977, 237917597],
    "scipy": [3510121, 34949961, 348316393, 3511474721, 34963718503],
}

numba_only = {"packed": []}

num_cores = [1, 2, 4, 8, 16, 32, 48]
core_scaling = {
    "cubes": [
        {"n": 1.1e8, "m": 100, "t": [0.0429]},
        {"n": 1.1e8, "m": 1e4, "t": [0.129]},
        {"n": 1.1e8, "m": 1e5, "t": [0.513]},
        {"n": 1.1e9, "m": 100, "t": [0.0539]},
        {"n": 1.1e9, "m": 1e4, "t": []},
        {"n": 1.1e9, "m": 1e5, "t": []},
    ]
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
    "search_cn": search_cn,
    "size": size,
    "n": n,
    "m": m,
    "marker": "o",
    "ls": "--",
    "extra": extra,
}

# %% [markdown]
# ## Thesan-XL
# IntCoordinates_111 from Flagship_new/PartType0 -
# $213209550877\sim 2.1\times10^{11}$ particles

# %%
n = []
m = [100, 1000, 10000, 100_000, 1_000_000, 10_000_000]

creation = {
    "data resetting": [],
    "python": [],
    "packed": [],
    "kdtree": [],
    "cubes": [],
    "scipy": [],
}

search = {
    "python": [],
    "packed": [],
    "pack_list": [],
    "kdtree": [],
    "cubes": [],
    "scipy": [],
}

search_cn = {
    "python": [],
    "packed": [],
    "pack_list": [],
    "kdtree": [],
    "cubes": [],
    "scipy": [],
}

size = {
    "dataset": [],
    "python": [],
    "packed": [],
    "scipy": [],
}

numba_only = {"packed": []}

num_cores = [1, 2, 4, 8, 16, 32, 48, 64]
core_scaling = {
    "cubes": [
        {"n": 1.1e8, "m": 100, "t": []},
        {"n": 1.1e8, "m": 1e4, "t": []},
        {"n": 1.1e8, "m": 1e5, "t": []},
        {"n": 1.1e9, "m": 100, "t": []},
        {"n": 1.1e9, "m": 1e4, "t": []},
        {"n": 1.1e9, "m": 1e5, "t": []},
    ]
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

ThesanXL = {
    "name": "LB_L10_CDM",
    "creation": creation,
    "search": search,
    "search_cn": search_cn,
    "size": size,
    "n": n,
    "m": m,
    "marker": "o",
    "ls": "--",
    "extra": extra,
}

# %% [markdown]
# ## Metadata

# %% jupyter={"source_hidden": true}
units = {
    "n": "particles",
    "creation": "s",
    "search": "ms",
    "search_cn": "ms",
    "size": "bytes",
}

# %% jupyter={"source_hidden": true}
metrics = {
    "creation": {
        "data resetting": "k",
        "packed": "tab:orange",
        "kdtree": "tab:olive",
        "cubes": "tab:purple",
        "scipy": "tab:green",
    },
    "search": {
        "packed": "tab:orange",
        "pack_list": "tab:brown",
        "kdtree": "tab:olive",
        "cubes": "tab:purple",
        "scipy": "tab:green",
    },
    "search_cn": {
        "brute": "k",
        "packed": "tab:orange",
        "pack_list": "tab:brown",
        "kdtree": "tab:olive",
        "cubes": "tab:purple",
        "scipy": "tab:green",
    },
    "size": {
        "dataset": "k",
        "python": "tab:blue",
        "packed": "tab:orange",
        "scipy": "tab:green",
    },
}


# %%
particle_threshold = 400

expected = {
    "creation": lambda n: n * np.log10(n),
    "search": lambda n: np.log10(n),
    # "search": lambda n: np.sqrt(n),
    # "size":lambda n: 1.2*n/10**(np.floor(np.log10(n/particle_threshold)))
    "search_cn": lambda n: np.log2(n),
    "size": lambda n: n,
}
expected_label = {
    "creation": r"$n\; \log(n)$",
    "search": r"$\log(n)$",
    # "search": r"$\sqrt{n}$",
    "search_cn": r"$\log{(n)}$",
    # "size":(
    #     r"$1.2\times\frac{n}{10^{\left\lfloor n/"
    #     f"{particle_threshold}"
    #     r"\right\rfloor}}$")
    "size": r"$n$",
}

m_expected = {
    "search_cn": lambda m: 1,
}
m_expected_label = {
    "search_cn": r""  # r"$m \times$"
}

ignore_metrics = ["creation:data resetting", "search_cn:brute"]

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
            y = np.array(sim[t].get(m, np.full_like(n, np.nan)))
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
            n = np.array(sim["n"])
            y = np.array(sim[t].get(m, np.full_like(n, np.nan)))
            msim = np.array(sim["m"])
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
                    (y / y[scale_index]) * expected[t](n[scale_index]) / expected[t](n)
                )
            axs[i].loglog(n, expected_y, color=c, marker=sim["marker"], ls=sim["ls"])
        axs[i].plot(np.nan, np.nan, color=c, label=m)

    axs[i].set_xlabel(f"n [{units['n']}]")
    axs[i].set_ylabel(f"{t}/{t}" r"$_0$ $/$ expected")
    label = expected_label[t]
    if used_m:
        label = m_expected_label[t] + label
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

for sim in sims:
    axs[0].plot(
        np.nan, np.nan, color="k", marker=sim["marker"], ls=sim["ls"], label=sim["name"]
    )
for ax in axs:
    ax.legend()

# %% [markdown]
# ## Normalized

# %% jupyter={"source_hidden": true}
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
            n = np.array(sim["n"])
            yk = np.array(sim[t]["scipy"])
            y = np.array(sim[t].get(m, np.full_like(yk, np.nan)))
            msim = np.array(sim["m"])
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

for sim in sims:
    axs[0].plot(
        np.nan, np.nan, color="k", marker=sim["marker"], ls=sim["ls"], label=sim["name"]
    )
for ax in axs:
    ax.legend()

# %% [markdown]
# ## Strong Scaling and Parallel Efficiency

# %%
fig, axs = plt.subplots(2, 1, sharex=True)
fig.set_figheight(10)
labels = {}
lw_base = 0.5
for sim in sims:
    if "num_cores" not in sim:
        continue
    num_cores = sim["num_cores"]
    core_scaling = sim["core_scaling"]["kdtree-search"]
    c = metrics["search_cn"]["kdtree"]
    for cs in core_scaling:
        n = cs["n"]
        m = cs["m"]
        t = np.array(cs["t"])
        label = f"n=$10^{{{int(np.log10(n))}}}$, m=$10^{{{int(np.log10(m))}}}$"
        if label not in labels:
            lw = lw_base
            axs[1].plot(np.nan, np.nan, color="k", label=label, lw=lw)
            lw_base *= 2
            labels[label] = lw
        else:
            lw = labels[label]
        axs[0].loglog(num_cores, t, color=c, marker=sim["marker"], ls=sim["ls"], lw=lw)
        t1 = t[0]
        efficiency = t1 / num_cores / t
        axs[1].semilogx(
            num_cores, efficiency, color=c, marker=sim["marker"], ls=sim["ls"], lw=lw
        )
    axs[0].plot(
        np.nan, np.nan, color="k", marker=sim["marker"], ls=sim["ls"], label=sim["name"]
    )


axs[0].set_ylabel("Strong Scaling [ms]")
axs[1].set_ylabel("Parallel Efficiency ($t_1/n/t_n$)")
axs[1].set_xlabel("Number of Cores")

axs[0].legend()
axs[1].legend()

# %% [markdown]
#

# %%
