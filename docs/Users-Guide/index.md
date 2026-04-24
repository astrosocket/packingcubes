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
##### From already loaded data
If you already have the data loaded in memory, the creation is trivial. 

In the following, assume `positions_data` is an `NxM` numpy matrix, where `M<=3`.

``` python
import packingcubes
cubes = packingcubes.Cubes(positions_data)
```

That's it, move on to [Searching](#searching-a-particlecubes-object).

Of course, there are a number of optional arguments that you can specify, along
with more complicated paths if you want to reuse information, save out data, etc.
See the [User's guide](Users-Guide/Tutorials/Creation) for more information.

##### From a snapshot
Creation is _slightly_ more complicated from a snapshot, since there's the potential
for multiple particle types. Here we'll assume we only want the gas particles
(`PartType0`) in an IllustrisTNG snapshot.

``` python
import packingcubes
cubes = packingcubes.Cubes(
    "path/to/IllustrisTNG/snapshot.hdf5", particle_type="PartType0"
)
```


#### Searching a ParticleCubes object
We'll assume we want to grab the particle indices in a sphere centered on `(500, 23, 7654.4)`, with radius `600`.

```python
index_array = cubes.get_indices_in_sphere(center, radius)
```

`index_array` will be an `Nx3` integer array where each row corresponds a chunk of
data in the form `[start, stop, partial]`. `start` and `stop` are the start/stop 
indices of the chunk (e.g. `data[start:stop]`) and `partial` denotes whether the
chunk was partially (`1`) or entirely (`0`) contained within the sphere.

For performance reasons, these data chunks may contain particles nearby the sphere, but not actually contained inside. If you want stricter control, see 
[get_particle_index_list_in_sphere](Users-Guide/API/packingcubes#packingcubes.ParticleCubes.get_particle_index_list_in_sphere).

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
    to save the sorted dataset information, see [Saving sorted datasets](#). Note
    that this does require som initial setup.

### Substituting SciPy's KDTree
--8<-- "README.md:KDTree"


### Command Line Interface

The command
``` bash
packcubes SNAPSHOT.hdf5 OUTPUT.hdf5
```

will generate the `ParticleCubes` data structure for each particle type found in the
`snapshot` file and store it in the `OUTPUT` file, creating if necessary. In this
context, a particle type is any top-level group in the snapshot whose name starts
with the string `Part`. See [The Command Line Interface](#) for more details.

