---
icon: lucide/hard-drive-download
---

# Installation

## Requirements
--8<---- "README.md:19:53"

## Installation is easy
<!-- --8<-- [start:basic-install] -->
We're on [PyPI](https://pypi.org/project/packingcubes):

=== "pip"

    ``` bash
    pip install packingcubes
    ```
    
=== "uv"
    ``` bash
    uv pip install packingcubes
    ```

=== "pixi"
    ``` bash
    pixi add packingcubes --pypi
    ```
<!-- --8<-- [end:basic-install] -->

Additional package requirements can be installed via optional dependencies
(see [Optional Packages](#optional-packages)). Example:

=== "pip"

    ``` bash
    pip install "packingcubes[viz, jupyter]"
    ```

=== "uv"

    ``` bash
    uv pip install packingcubes --extra viz --extra jupyter
    ```

=== "pixi"

    ``` bash
    pixi add packingcubes[viz, jupyter] --pypi
    ```
