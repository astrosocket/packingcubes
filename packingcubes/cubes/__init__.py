"""
Package for creating and manipulating packing cubes

Can be used via CLI or programmatically. The CLI essentially calls
`make_cubes` and then `Cubes` on the output of whatever snapshot it's pointed
at, after which it saves the generated cubes structure.

`Cubes` is intended to be the primary entry point when used programmatically,
calling `make_cubes` under the hood as needed. It will return either a
`ParticleCubes` or `dict[str, ParticleCubes]` object, depending on the number
particle types found.

MultiCubes is essentially identical to ParticleCubes, with each method
having one additional parameter, `particle_types`, that is used to select which
particle type information to return.
The output of each method differs as well. If `output` would be the output from
a ParticleCubes method call,
```python
{"PartType0":output_for_PartType0, "PartType1":output_for_PartType1, ...}
```
would be the MultiCubes output.

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
