---
icon: lucide/package-search
---

# Finding Particles Within a Shape


We have 4 goals for this tutorial:

1. Create some particle position data
2. Create a `ParticleCubes` object
3. Find all the particles in a square
4. Find all the particles within a circle

We'll assume you already are somewhat familiar with `numpy` and `matplotlib`.

We'll keep everything in 2D to make plotting easier, but all of these steps 
can be done in 1D & 3D as well[^1].  

[^1]: The `packingcubes` portions are actually much simpler in 3D, as will 
    become obvious shortly, but the plotting becomes a lot harder.


## Install and Import Dependencies


Since we'll be using `matplotlib` for visualization, we'll want to include the
`viz` optional dependencies:

=== "pip"
    ``` bash
    $ pip install "packingcubes[viz]
    ```

=== "pixi"
    ``` bash
    $ pixi add packingcubes[viz] --pypi
    ```


Now import the modules we need

```python
import numpy as np # for generating data
import matplotlib.pyplot as plt

import packingcubes
```

## Create positions data


We'll start by generating some random data. We'll make 1000 particles with $x$ and $y$
coordinates ranging from 0 to 100.

```python
xy = np.random.uniform(size=(1000,2)) * 100

plt.plot(xy[:,0], xy[:,1] , "k.")
```

ParticleCubes are designed to work with 3D data, so we'll need to pad our 2D
data with zeros to make it 3D.

```python
positions = np.zeros_like(xy, shape=(len(xy),3))
positions[:, :2] = xy
```

??? tip
    [OpTrees](../Reference/Packed-Trees#Optree) will do this padding for you
    at the cost of being a little more opaque, and *only* working with position
    data.

## Create a ParticleCubes object


We don't need anything fancy, so creating a `ParticleCubes` is pretty simple:

```python
cubes = packingcubes.Cubes(dataset=positions, particle_threshold=10)
cubes
```

## Find all particles in a square


### Define our square


We'll look at the square whose bottom-left corner is $(20, 21)$ and that has a side-length of 10.

```python
plt.plot(xy[:,0], xy[:,1] , "k.")
bx = 20
by = 21
side = 20
plt.plot(bx + np.array([0, side, side, 0, 0]), by + np.array([0, 0, side, side, 0]), lw=2)
```

To search in a box, we set the corner position and then the dimensions of the
box as a single array in the form `[x, y, z, dx, dy, dz]`.

Unfortunately, `ParticleCubes` do not currently support 1D or 2D search shapes.
Luckily, making our square 3D is easy, just set the $z$ position to 0:

```python
box = [bx, by, 0, side, side, side]
```

Having 3D search volume with 2D data is fine, `packingcubes` will effectively
just ignore the third dimension. You just need to ensure the box actually has
a volume. Setting `dz=0` would raise an error.


### Actually do the search

```python
index_array = cubes.get_particle_indices_in_box(box)
```

Index array is an array of data chunk indices, where each row represents a
data chunk in the form `[start, stop, partial]`. `partial` just specifies if
the entire chunk is contained (`0`) or if it's only _partially_ contained (`1`).

```python
plt.plot(xy[:,0], xy[:,1] , "k.")
plt.plot(bx + np.array([0, side, side, 0, 0]), by + np.array([0, 0, side, side, 0]), lw=2)

for start, stop, partial in index_array:
    # plot each chunk of data
    chunk = positions[start:stop, :2]
    plt.plot(chunk[:,0], chunk[:, 1], "*" if partial else "o")
```

## Find all particles in a circle


### Define the circle


We'll look at the circle centered at $(64, 73)$ with radius $30$. 

Note that this circle extends outside our data bounds!

We'll need to do the same `z=0` trick when converting to a 3D sphere.

```python
center = [64, 73, 0]
radius = 30
```

```python
plt.plot(xy[:,0], xy[:,1] , "k.")
circle = plt.Circle(center[:2], radius, color='tab:blue', lw=2, clip_on=False, fill=False)
plt.gca().add_patch(circle)
```

### Actually do the search


This works pretty much identically to the square (box):

```python
index_array = cubes.get_particle_indices_in_sphere(center=center, radius=radius)
```

```python
plt.plot(xy[:,0], xy[:,1] , "k.")
circle = plt.Circle(center[:2], radius, color='tab:blue', lw=2, clip_on=False, fill=False)
plt.gca().add_patch(circle)

for start, stop, partial in index_array:
    # plot each chunk of data
    chunk = positions[start:stop, :2]
    plt.plot(chunk[:,0], chunk[:, 1], "*" if partial else "o")

```

!!! warning "Fragile Data!"
    You may notice that the order of the data in `positions` has changed (occurred
    when we made `cubes`). This is by design! But it also means if you modify 
    `positions` you will break the linkage between search results and the data.
    For more on this, see [Working with Datasets](../Recipes/Working_with_datasets)
    for more robust ways to interact with data.


<script id="MathJax-script" src="https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };

  document$.subscribe(() => {
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })
</script>
