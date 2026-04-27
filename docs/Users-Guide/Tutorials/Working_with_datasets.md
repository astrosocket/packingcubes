---
icon: lucide/grid-3x2
---

We have 4 main goals for this tutorial:

1. Create a Dataset
    1. From data in memory
    2. From a snapshot
2. Run `packingcubes` on the dataset
3. Save the sorted dataset
4. Load the dataset

For the tutorial, you'll need 


<div class="grid cards" markdown>
    
- **Access to a Snapshot**
        
    ---
    You'll need access to a snaphsot file that looks GADGETish. So anything from
    Gadget (v2 and up),
    [Gizmo](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html),
    [Swift](https://swift.strw.leidenuniv.nl/docs/index.html), 
    [OpenGadget](https://www.space-coe.eu/codes/opengadget.php), 
    [Arepo](https://arepo-code.org/), and similar should work.

    We'll be using a snapshot from IllustrisTNG, based on Arepo, for this
    tutorial.

</div>

??? note "Don't have a snapshot?"
    If you don't have a snapshot readily available, we will be using 
    [snapshot_090.hdf5](https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/L25n256/CV/CV_3/snapshot_090.hdf5)
    (about $10^7$ particles, 2.5 GB) from one of the IllustrisTNG runs
    used in the
    [CAMELS](https://camels.readthedocs.io/en/latest/index.html) project.

## Install and Import Dependencies


As always for a tutorial, we'll need to install and import the dependencies
we'll need:

=== "pip"
    ``` bash
    $ pip install packingcubes
    ```

=== "pixi"
    ``` bash
    $ pixi add packingcubes --pypi
    ```


Now import the modules we need

```python
import numpy as np 

import packingcubes
```

## Creating a Dataset


The first thing we need to do for anything with `packingcubes` that will be
used long term is create a `Dataset` object. We'll demonstrate both of the 
primary ways to do that here.


### From a Snapshot


Creating a dataset from a snapshot is pretty straightforward if it's Gadget-based[^1]. 

[^1]: Follows the Gadget-2 header specification 
    [here](https://wwwmpa.mpa-garching.mpg.de/gadget/html/structio__header.html)

```python
dataset = packingcubes.GadgetishHDF5Dataset(filepath="./snapshot_090.hdf5")
```

We need to choose which particles we want to look at

```python
dataset.particle_type = "PartType0" # the gas particles
```

Then we can check the particle positions are loaded:

```python
dataset.positions
```

??? tip "Eager Loading"
    `GadgetishHDF5Datasets` eagerly load `positions` data, so you can specify
    which particle type you want to load first by setting the
    `initial_particle_type` parameter in the constructor.

### From positions in memory


??? question "Déjà-vu?"
    This part will actually be very similar to the 
    [:lucide-package-search: Finding Particles Within a Shape](ParticlesWithinShape)
    tutorial, because `Cubes` does this step internally when you pass it an array.


We'll start by generating some random data. We'll make 1000 particles with
coordinates ranging from 0 to 100.

```python
positions = np.random.uniform(size=(1000,3)) * 100
```

Then to create a dataset, just use

```python
inmem_dataset = packingcubes.InMemory(positions=positions)
```

## Run `packingcubes`


We now want to cubify our datasets.

It's actually the same command (`Cubes`), regardless of which kind of dataset
you're using. However, we'll need to specify that we only want the gas particles
at this time for the GadgetishHDF5Dataset[^2][^3].

[^2]: Otherwise you'll get whatever particles are currently loaded (i.e.
    whatever is in `dataset.particle_type`). By default, for
    `GadgetishHDF5Dataset`s, that's the first top-level group whose name
    starts with `"Part"`, and the first element of `dataset.particle_types`. 
[^3]: We don't need to specify the particle type for InMemory datasets because
    they only have one, `"PartTypeIM"`. This is just a dummy name used when
    saving the particles out, however; you can change it with 
    `inmem_dataset.particle_type = "NewName"`. Note that the new name would
    need to start with `"Part"` for it to be picked up by 
    `GadgetishHDF5Dataset`.

Note that the first time you run `Cubes`, it will take a little bit to 
Just-In-Time compile some of the functions. Subsequent invocations will be
faster.

```python
cubes = packingcubes.Cubes(dataset=dataset, particle_type="PartType0")
inmem_cubes = packingcubes.Cubes(dataset=inmem_dataset)
```

These cubes are ready for searching!

```python
cubes.get_particle_indices_in_sphere(
    center=[13548,23147,1684], radius=1000,
)
```

```python
inmem_cubes.get_particle_indices_in_sphere(
    center=[64,73,15], radius=20,
)
```

## Saving a Dataset


Generally, you'll only want to do the initial cubing once and save the
results. (Though it is fast enough you could regenerate it each time for
"small" datasets).

??? info "Saving Cubes recap"
    You likely have already seen how to save the cubes information in the 
    [Getting Started](../../index.md) page, but as a recap, it's just
    `cubes.save("snapshot_cubes.hdf5")`

Saving the cubes structure doesn't save the sorted particle position
information to disk. You'll need to separately save it:

```python
dataset.save(output_file="sorted_positions.hdf5")
inmem_dataset.save(output_file="inmem_positions.hdf5")
```

## Loading a sorted dataset


If you have an already sorted dataset, like we now do, you have two options:

1. Specify the sorted dataset as your dataset filepath - Simpler, but will only
   contain the fields you've already sorted (so just `positions`)
2. Pass the sorted dataset to the `sorted_filepath` parameter - An extra step, but
   will check the sorted dataset for any fields first before loading from the
   original.

??? tip
    You can also pass the `sorted_filepath` parameter to the `Cubes` call
    directly!

We'll pick option 2.  

```python
dataset_reloaded = packingcubes.GadgetishHDF5Dataset(
    filepath="snapshot_090.hdf5",
    sorted_filepath="sorted_positions.hdf5",
)
```

```python
dataset_reloaded.positions
```

Note the sorted positions!



## All-in-one
As mentioned previously, the `Cubes` command combines a number of the above
steps into one command. So if you don't need any additional flexibility, or
access to the dataset positions/shuffle list/etc., you can include the dataset
saving in the initial `Cubes` call via the `sorted_filepath` and `save_dataset`
parameters, like so:

```python
cubes = packingcubes.Cubes(
    dataset="./snapshot_090.hdf5", 
    particle_type="PartType0",
    sorted_filepath="sorted_positions.hdf5",
    save_dataset=True,
)
```

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
