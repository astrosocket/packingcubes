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
import matplotlib as mpl
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
    "data resetting": [1.13e-3, 0.012, 0.127],
    "python": [0.0482, 0.54, 6.49],
    "packed": [0.00275, 0.0337, 0.360],
    "kdtree": [0.005, 0.039, 0.343],
    "cubes": [0.007, 0.038, 0.346],
    "scipy": [0.0208, 0.300, 4.33],
}

search = {
    "python": [1.25, 4.49, 17.8],
    "packed": [0.0329, 0.102, 0.3811],
    "pack_list": [0.0418, 0.1849, 2.3762],
    "kdtree": [0.0417, 0.0832, 2.3758],
    "cubes": [0.027, 0.0392, 0.0911],
    "scipy": [0.118, 1.66, 20.1],
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

query = {
    "kdtree": [0.0862, 0.1315, 0.2511],
    "scipy": [0.0078, 0.0087, 0.0085],
}

numba_only = {"packed": [0.0315, 0.0299, 0.0402]}

num_cores = [1, 2, 4, 8, 16]
core_scaling = {
    "cubes-create": [{"n": 1.6e7, "t": [0.695, 0.407, 0.336, 0.365, 0.348]}],
    "kdtree-create": [{"n": 1.6e7, "t": [0.445, 0.267, 0.185, 0.210, 0.186]}],
    "cubes-search": [
        {"n": 1.6e7, "m": 100, "t": [0.0207, 0.021, 0.022, 0.037, 0.083]},
        {"n": 1.6e7, "m": 1e5, "t": [0.305, 0.283, 0.266, 0.335, 0.269]},
        {"n": 1.6e7, "m": 1e6, "t": [1.02, 0.872, 0.715, 1.13, 1.46]},
    ],
    "kdtree-search": [
        {"n": 1.6e6, "m": 1e4, "t": [0.140, 0.101, 0.086, 0.114, 0.099]},
        {"n": 1.6e7, "m": 100, "t": [0.0242, 0.0248, 0.025, 0.034, 0.039]},
        {"n": 1.6e7, "m": 1e5, "t": [0.597, 0.451, 0.387, 0.442, 0.434]},
        {"n": 1.6e7, "m": 1e6, "t": [3.046, 2.140, 1.802, 1.772, 1.787]},
    ],
}

size = {
    "dataset": [1871217, 18710649, 187104885],
    "python": [545461, 5231664, 50810413],
    "packed": [38088, 359177, 3468337],
    "scipy": [5063559, 50484807, 508383527],
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
    "data resetting": [1.15e-3, 12.3e-3, 134e-3],
    "python": [0.0482, 0.575, 6.65],
    "packed": [0.0379, 0.460, 5.53],
    "kdtree": [np.nan, np.nan, np.nan],
    "cubes": [np.nan, np.nan, np.nan],
    "scipy": [0.0186, 0.231, 2.82],
}

search = {
    "python": [1.26, 4.87, 20.3],
    "packed": [0.87, 2.07, 4.66],
    "pack_list": [np.nan, np.nan, np.nan],
    "kdtree": [np.nan, np.nan, np.nan],
    "cubes": [np.nan, np.nan, np.nan],
    "scipy": [0.523, 3.68, 32.9],
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
    "dataset": [1909377, 19092225, 190920693],
    "python": [475785, 4987588, 50087794],
    "packed": [33288, 342857, 3423397],
    "scipy": [5165319, 51502343, 518559015],
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
    "data resetting": [1.22e-3, 14.3e-3, 142e-3],
    "python": [0.050, 0.578, 6.59],
    "packed": [0.0405, 0.463, 5.42],
    "kdtree": [np.nan, np.nan, np.nan],
    "cubes": [np.nan, np.nan, np.nan],
    "scipy": [0.0186, 0.244, 3.05],
}

search = {
    "python": [3.21, 12.2, 52.2],
    "packed": [1.7, 4.68, 12.2],
    "pack_list": [np.nan, np.nan, np.nan],
    "kdtree": [np.nan, np.nan, np.nan],
    "cubes": [np.nan, np.nan, np.nan],
    "scipy": [1.06, 8.02, 90.2],
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
    "dataset": [3813237, 38130837, 381306861],
    "python": [545559, 4974569, 48502428],
    "packed": [38088, 341617, 3312897],
    "scipy": [5158217, 51431113, 517846503],
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
    "packed": [4.555e-03, 5.249e-02, 7.246e-01, 8.161e00, 9.263e01],
    "kdtree": [1.394e-02, 4.127e-02, 3.901e-01, 3.833e00, 3.858e01],
    "cubes": [1.388e-02, 4.104e-02, 3.905e-01, 3.754e00, 3.852e01],
    "scipy": [0.0264, 0.303, 3.77, 46.6, 561],
}

search = {
    "python": [2.5, 8.7, 61, 208, 850],
    "packed": [0.0784, 0.2182, 0.8734, 3.6768, 16.3549],
    "pack_list": [1.3838e-01, 4.2616e-01, 2.5076e00, 2.1078e01, 1.7926e02],
    "kdtree": [0.2755, 0.5671, 2.9058, 26.796, 243.3203],
    "cubes": [0.165, 0.3703, 1.0345, 3.4427, 64.9992],
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
        [0.053, 0.0658, 0.1215, 0.0438, 0.1474, 0.118],
        [0.0658, 0.0829, 0.1938, 0.4717, 0.0734, 0.5635],
        [0.0656, 0.1025, 0.1854, 0.6704, 2.3163, 0.2082],
        [0.0701, 0.0924, 0.1888, 0.6229, 2.5152, 10.3029],
        [0.072, 0.1019, 0.2273, 0.7805, 2.9204, 11.6037],
    ],
    "pack_list": [
        [0.0741, 0.1023, 0.2347, 0.151, 0.3363, 0.28],
        [0.0822, 0.1142, 0.3045, 0.9416, 0.924, 1.7198],
        [0.0893, 0.1341, 0.3229, 1.2564, 4.9119, 24.3137],
        [0.0948, 0.1341, 0.3252, 1.1672, 5.243, 40.6299],
        [0.0975, 0.1473, 0.3883, 1.3552, 5.9009, 44.5103],
    ],
    "kdtree": [
        [0.2021, 0.2374, 0.3192, 0.4334, np.nan, np.nan],
        [0.2093, 0.2571, 0.4053, 0.9141, 1.9314, np.nan],
        [0.2152, 0.2649, 0.4056, 1.2328, 4.4435, 35.0331],
        [0.2117, 0.2567, 0.4153, 1.127, 4.5874, 47.3734],
        [0.2367, 0.2654, 0.4371, 1.1882, 5.5671, 48.2764],
    ],
    "cubes": [
        [0.139, 0.1707, 0.2142, 0.2062, 0.2716, 0.2621],
        [0.1526, 0.177, 0.2862, 0.6274, 0.2362, 0.6893],
        [0.1617, 0.1841, 0.2747, 0.9181, 2.1999, 0.4072],
        [0.1573, 0.1885, 0.2861, 0.8052, 2.9047, 10.0253],
        [51.4001, 0.212, 0.3387, 0.865, 3.2409, 13.6392],
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
    "kdtree-create": [
        {"n": 1.1e9, "t": [83.48, 43.01, 22.01, 12.32, 7.35, 3.91, 4.399]}
    ],
    "kdtree-search": [
        {"n": 1.1e8, "m": 1e4, "t": [0.422, 0.353, 0.341, 0.342, 0.352, 0.382, 0.421]},
        {"n": 1.1e9, "m": 100, "t": [0.102, 0.105, 0.115, 0.125, 0.141, 0.191, 0.215]},
        {"n": 1.1e9, "m": 1e4, "t": [1.57, 1.33, 1.23, 1.21, 1.22, 1.12, 1.17]},
        {"n": 1.1e9, "m": 1e5, "t": [7.35, 6.07, 5.69, 5.46, 5.38, 5.26, 5.04]},
    ],
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
    "num_cores": num_cores,
    "core_scaling": core_scaling,
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
        {"n": 2.1e8, "m": 1e4, "t": [0.161]},
        {"n": 2.1e9, "m": 100, "t": []},
        {"n": 2.1e9, "m": 1e5, "t": []},
        {"n": 2.1e9, "m": 1e6, "t": []},
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
colors = {
    "IllustrisTNG": "tab:blue",
    "LB_L10_CDM": "tab:orange",
    "ThesanXL": "tab:green",
}
test = "kdtree-create"
for sim in sims:
    if "num_cores" not in sim:
        continue
    num_cores = sim["num_cores"]
    if test not in sim["core_scaling"]:
        continue
    core_scaling = sim["core_scaling"][test]
    c = colors.get(sim["name"], "tab:gray")
    for cs in core_scaling:
        n = cs["n"]
        t = np.array(cs["t"])
        label = f"n=$10^{{{int(np.log10(n))}}}$"
        if label not in labels:
            lw = lw_base
            axs[1].plot(np.nan, np.nan, color="k", label=label, lw=lw)
            lw_base += 1
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
        np.nan, np.nan, color=c, marker=sim["marker"], ls=sim["ls"], label=sim["name"]
    )

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

# %% [markdown]
#

# %%
