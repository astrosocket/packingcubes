"""
Package for creating and manipulating packing cubes

Can be used via CLI or programmatically. The CLI essentially calls
`make_cubes` and then `Cubes` on the output of whatever snapshot it's pointed
at, after which it saves the generated cubes structure. Use the `--help`
argument for more information.

`Cubes` is intended to be the primary entry point when used programmatically,
calling `make_cubes` under the hood as needed. It will return a
`ParticleCubes` object.

`ParticleCubes` are the main workhorse; construction via `make_cubes` does the
initial dataset sorting, while each `ParticleCubes` object has a number of
search methods attached, like `get_particles_in_sphere`.

MultiCubes are essentially identical to ParticleCubes, with each method
having one additional parameter, `particle_types`, that is used to select which
particle type information to return.
The output of each method differs as well. If `output` would be the output from
a ParticleCubes method call,
```python
{"PartType0":output_for_PartType0, "PartType1":output_for_PartType1, ...}
```
would be the MultiCubes output.  Effectively, it acts as a dictionary-based
wrapper around a collection of ParticleCubes objects, and would be used in the
case when you want some or all particle types from a search, not just one.

Classes
-------
ParticleCubes
    Object to perform rapid, parallel searches of a dataset
MultiCubes
    Collection of ParticleCubes organized by particle type

Functions
---------
Cubes
    Load or create a ParticleCubes object (calls `make_cubes` under the hood as
    needed)
make_cubes
    The actual ParticleCubes creation
make_MultiCubes
    Like Cubes but explicitly creates a MultiCubes object, even if only one
    particle type is present

"""

from packingcubes.cubes.cubes_creation import (
    Cubes as Cubes,
)
from packingcubes.cubes.cubes_creation import (
    CubesError as CubesError,
)
from packingcubes.cubes.cubes_creation import (
    make_cubes as make_cubes,
)
from packingcubes.cubes.cubes_creation import make_MultiCubes as make_MultiCubes
from packingcubes.cubes.multi_cubes import MultiCubes as MultiCubes
from packingcubes.cubes.particle_cubes import ParticleCubes as ParticleCubes

__all__ = [
    "Cubes",
    "CubesError",
    "make_cubes",
    "MultiCubes",
    "ParticleCubes",
    "make_MultiCubes",
]
