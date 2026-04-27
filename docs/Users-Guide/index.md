---
icon: lucide/rocket
---
# Getting Started

--8<--- "docs/Installation.md:basic-install"




## Basic Usage

### Searching Data
The steps for searching data are pretty straightforward:

1. Create ParticleCubes object(s)
2. Search object

#### Creating a ParticleCubes object
Basic steps to creating a `ParticleCubes` object.

Of course, there are a number of optional arguments that you can specify, along
with more complicated paths if you want to reuse information, save out data, etc.
See the [tutorials](Tutorials) and [package overview](Reference) for more information.

##### From already loaded data
If you already have the data loaded in memory, creation is trivial. 

In the following, assume `positions_data` is an `Nx3` numpy matrix.

``` python
import packingcubes
cubes = packingcubes.Cubes(positions_data)
```

That's it, move on to [Searching](#searching-a-particlecubes-object).

??? info "1D/2D and ND matrices"
    For technical reasons, direct `ParticleCubes` creation from 1D and 2D
    matrices are not supported. See 
    [:lucide-package-search: Finding Particles Within a Shape](Tutorials/ParticlesWithinShape#create-positions-data)
    for an example of converting 2D data to 3D. 4D and higher dimensional
    matrices are not supported.


##### From a snapshot
Creation is _slightly_ more complicated from a snapshot, since there's the potential
for multiple particle types. Here we'll assume we want the gas particles
(`PartType0`) in an IllustrisTNG snapshot.

``` python
import packingcubes
cubes = packingcubes.Cubes(
    "path/to/IllustrisTNG/snapshot.hdf5", particle_type="PartType0"
)
```


#### Searching a ParticleCubes object
Let's say we want to grab the particle indices in a sphere centered on 
`(500, 23, 7654.4)`, with radius `600`.

```python
center = [500, 23, 7654.4]
radius = 600
index_array = cubes.get_indices_in_sphere(center, radius)
```

`index_array` will be an `Nx3` integer array where each row corresponds to a chunk of
data in the form `[start, stop, partial]`. `start` and `stop` are the start/stop 
indices of the chunk (e.g. `data[start:stop]`) and `partial` denotes whether the
chunk was partially (`1`) or entirely (`0`) contained within the sphere.

For performance reasons, these data chunks may contain particles nearby the sphere,
but not actually contained inside (for an example, see 
[:lucide-package-search: Finding Particles within a Shape](Tutorials/ParticlesWithinShape#actually-do-the-search_1)).
If you want stricter control, see 
[get_particle_index_list_in_sphere](Reference#packingcubes.ParticleCubes.get_particle_index_list_in_sphere).

??? note "Data wrapping"
    `packingcubes` does not (yet!) support toroidal topology for snapshots. Meaning 
    if the snapshot has `x` bounds `[0, 100]`, a sphere with radius `10` centered
    at `[0, 0, 0]` would only search the region from 0 to 10 and will not include
    the region from 90 to 100. See 
    [Issue #28](https://github.com/astrosocket/packingcubes/issues/28).

??? warning "Units"
    `packingcubes` does not use units internally. That means any quantities defining
    a search volume must be in the same units as the positions data (including
    hubble parameters, if present). You can use quantities with units, e.g.
    `center = [10, 10, 10] * kpc`, but no conversion occurs if the positions data
    doesn't match (e.g. if it is in `Mpc/h`). See 
    [Issue #21](https://github.com/astrosocket/packingcubes/issues/21).

#### Saving a ParticleCubes object
Saving is also simple:
``` python
cubes.save("path/to/output.hdf5")
```

Loading it is identical to loading a snapshot:
``` python
cubes = packingcubes.Cubes("path/to/output.hdf5", particle_type="gas")
```

??? info "Data saving"
    Saving cubes information **only** saves information about the cubes. If you want
    to save the sorted dataset information, specify the `save_dataset=True`
    and the `sorted_filepath="path"` option on the initial `Cubes` call or see 
    [Saving sorted datasets](Tutorials/Working_with_datasets#saving-a-dataset). Note
    that while this second option does require more initial setup, it's more flexible.

#### Specifying Number of Threads
Both the `ParticleCubes` creation and search operate in parallel using `numba`'s
[threading layers](https://numba.readthedocs.io/en/stable/user/threading-layer.html).

By default we automatically use all threads available, but you can set a lower
number (if e.g. your code also does work in parallel) by either setting the
`NUMBA_NUM_THREADS` environment variable or `#!python numba.set_num_threads(num_threads)`, 
where `num_threads <= numba.config.NUMBA_NUM_THREADS`.

### Substituting SciPy's KDTree
--8<-- "README.md:KDTree"

More information can be found in the 
[Optrees Documentation](Reference/Packed-Trees#OpTree).

### Command Line Interface

The command
``` bash
packcubes SNAPSHOT.hdf5 OUTPUT.hdf5
```

will generate the `ParticleCubes` data structure for each particle type found in the
`snapshot` file and store it in the `OUTPUT` file, creating if necessary. In this
context, a particle type is any top-level group in the snapshot whose name starts
with the string `Part`. See [The Command Line Interface](Reference/CLI) for more details.

