[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)

# packingcubes

Compact oct-tree implementation used for Socket

## Development requirements
### uv
To get ready for development, create a virtual enviroment and install the package:
```
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

We use ruff for formatting. When you go to commit your code, it will automatically be formatted thanks to the pre-commit hook.

Tests are performed using pytest.
### pixi
Using with pixi is pretty easy, simply
```sh
pixi shell
```
To look at visualizations, run tests, or develop, simply specify the corresponding environment
```sh
# visualizations
pixi shell -e viz
# testing (also includes viz)
pixi shell -e test
# developing (also includes viz & test)
pixi shell -e dev
```
and e.g. to run tests, say
```sh
pixi run test
```
which runs `pytest  --cov=packingcubes`.
