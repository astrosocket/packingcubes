---
icon: lucide/rocket
---
# Getting Started

--8<--- "docs/Installation.md:basic-install"




## Basic Usage

### Searching Data

If you just want to find the values of a particular field in a spherical region
of your dataset, the simplest way to do it is the following "one-liner":

```python
import packingcubes

field_in_sph = packingcubes.Cubes(
    positions_array, # (1)!
    extras={"field_name":field_array} # (2)!
).Sphere(center, radius, fields="all").field_name
```

1. `positions_array` is a 3D array of particle position data
2. `field_array` is an array with `N` or `NxM` elements, where `N` is the
    number of particles. `"field_name"` can be whatever you want to later call it.

That's it.


??? tip "Positions Only?"
    Position data is always included, so if that's all you need, the "one-liner"
    is even shorter: 
    ```python
    pos_in_sph = packingcubes.Cubes(positions_array).Sphere(center, radius).positions
    ```
    
??? info "1D/2D and ND matrices"
    For technical reasons, direct `ParticleCubes` creation from 1D and 2D
    matrices are not supported. See 
    [:lucide-package-search: Finding Particles Within a Shape](Tutorials/ParticlesWithinShape#create-positions-data)
    for an example of converting 2D data to 3D. 4D and higher dimensional
    matrices are not supported.

In the next couple sections, we'll briefly explain what each step does, but for
more information, see the [:lucide-info: Tutorials](Tutorials),
[:lucide-cooking-pot: Recipes](Recipes), and [:lucide-library: Reference](Reference)
sections.

#### Creating a Dataset and ParticleCubes object
The `packingcubes.Cubes(...)` function as used here combines three steps into one:

1. Create Dataset object (an
   [InMemory](Reference/Data-Objects#packingcubes.data_objects.InMemory) object
   in this case)
2. Sort the positions and create a
   [ParticleCubes](Reference/Cubes#packingcubes.cubes.ParticleCubes) object
3. Sort the extra field `field_array` and attach it to the Dataset.

[`Dataset`](Reference/Data-Objects)s are multifunctional objects whose main
purpose is to unify various forms of positions data in a common format (as
`dataset.positions`). It's also used to sort any extra fields via the "shuffle
list" (`dataset.index`), an array containing the orginal index for each sorted
position. In this example, it acts as thin wrapper on the array, but it can
also be used for loading entire [snapshots](Tutorials/Working_with_datasets).

[`ParticleCubes`](Reference/Cubes#packingcubes.cubes.ParticleCubes) are the
main workhorses of `packingcubes`; they contain the cubed-octree
structure/metadata that produces all the cool features. Constructing a
`ParticleCubes` object performs the initial sorting of the positions data
(in parallel!) and creates the "cubes" and 
["octrees"](Reference/#packingcubes.PackedTree). 

#### Searching the ParticleCubes
The next part, [`.Sphere(...)`](Reference/#packingcubes.ParticleCubes.Sphere), searches the ParticleCubes object and finds
all particles contained within the specified spherical region. It then creates
a subdataset containing the particle positions and any additional fields
(specified with the `fields` parameter, `"all"` includes everything available
in `dataset.extras`) corresponding to solely those particles that are contained
(but see note about Extra Particles).


??? info "Extra particles"
    By default, searches on `ParticleCubes` are assumed to be not _strict_ for
    maximal performance. So particle containment is tested on an octree node
    level, i.e. if the node overlaps the containment region, all particles in
    the node are considered to be contained. This may increase the number
    of returned particles by a small fraction. If this is unacceptable, several
    of the search methods, including `Sphere` and `Box`, allow passing a
    `strict` parameter, which forces `packingcubes` to test each position in
    nodes that only partially overlap for containment. This imposes an
    additional, usually minor, performance penalty.
    
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

Lastly, we access the specified field attribute of our new subdataset.

#### Multiline approach
The "one-liner" above can be broken up into multiple lines to better highlight
the object pipeline:

```python
dataset = packingcubes.InMemory(positions=positions_array)
cubes = packingcubes.Cubes(dataset)
dataset.process_extra_fields({"field_name":field_array})
sphere = cubes.Sphere(center, radius, fields="field_name")
field_in_sph = sphere.field_name
```


### Saving a Dataset/ParticleCubes
Saving is simple, though it does break up the "one-liner":
``` python
cubes = packingcubes.Cubes(
    positions_array,
    save_dataset=True, # (1)!
    sorted_filepath="path/to/output.hdf5",
    extras={"field_name":field_array}
)
cubes.save("cubes_output.hdf5") # (2)!
sphere = cubes.Sphere(
    center, radius,
    save_filepath="sphere_output.hdf5", # (3)!
    fields="all",
)
```

1. Saves the sorted dataset (including anything in `extras`) to the
   `sorted_filepath`.
2. Recommend using the same file as `sorted_filepath`.
3. Should **not** be the same as `sorted_filepath`, unless you *want* to
   overwrite everything...

Loading it is identical to loading a snapshot:
``` python
cubes = packingcubes.Cubes("path/to/output.hdf5", particle_type="gas")
```

### Specifying Number of Threads
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

