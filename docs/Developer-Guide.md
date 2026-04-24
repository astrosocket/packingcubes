---
icon: lucide/square-code
---

Code contributions are very welcome! 

We recommend starting with [setting up a development environment](#dev-environment).

## Dev Environment
To setup the code for development, first clone the latest master from the repository:
``` bash
git clone https://github.com/astrosocket/packingcubes.git
```
then install as an editable:

=== "pip"
    ```bash
    cd packingcubes
    pip install -e .[dev, docs]
    ```

=== "pixi"

    ``` bash
    cd packingcubes
    pixi install -e dev
    ```

    !!! note
    You don't need to run `pixi install` if you are planning on running tests 
    (`pixi run test`), building documentation (`pixi run zserve`/`pixi run zbuild`),
    or working in the shell (via `pixi shell`), since those commands will install the
    relevant environment under the hood.

and then install the pre-commit hooks: `pre-commit install`.


## Contributing

--8<--- "CONTRIBUTING.md:contributing"

--8<--- "CONTRIBUTING.md:dev-tools"

## Documentation

!!! warning
    The documentation building tools are in the `docs` dependency group
    and are **not** part of `dev`!

--8<--- "CONTRIBUTING.md:documentation"
