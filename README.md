# packingcubes

Compact oct-tree implementation used for Socket

## Development requirements

To get ready for development, create a virtual enviroment and install the package:
```
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

We use ruff for formatting. When you go to commit your code, it will automatically be formatted thanks to the pre-commit hook.

Tests are performed using pytest.
