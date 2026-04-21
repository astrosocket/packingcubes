# Contributing to packingcubes


Contributions for `packingcubes` should come as pull requests submitted through our [GitHub repository](https://github.com/astrosocket/packingcubes).

Contributions are always welcome, but you should make sure of the following:

+ Your contributions pass all automated tests (you can check this with `pytest`).
+ Your contributions add tests for new functionality (and the tests pass!).
+ Your contributions are formatted with the `ruff` formatter.
+ Your contributions pass `ruff` style checks.
+ Your contributions are documented with docstrings and their style passes the `ruff` checks.
+ Your contributions are documented fully under `/docs`.

You should also abide by the [code of conduct](https://github.com/astrosocket/packingcubes/CODE_OF_CONDUCT.md).

Some brief quickstart-style notes are included below, but are not intended to replace consulting the documentation of each relevant toolset. We recognize that this can seem daunting to users new to collaborative development. Don't hesitate to get in touch for help if you want to contribute!

## Installing the toolkit
The tools in the following sections are all included in the `dev` dependency group, installed
as explained in the [Installation](docs/Installation.md) section of the website. Some tools may
work as is; others may require some additional setup as explained in that section.

While you may use any dev environment you wish, much of the tooling is specified 
using `pixi`, so we recommend using that (as opposed to e.g. `pip + venv` or 
analogous). 


## Ruff

To check that your copy of the repository conforms to style rules you can run `ruff check` in the same directory as the `pyproject.toml` file. A message like `All tests passed!` indicates that your working copy passes the checks, otherwise a list of problems is given. Some might be automatically fixable with `ruff check --fix`. Don't forget to commit any automatic fixes.

`ruff` is also used to enforce code formatting, you can check this with `ruff format --check` and automatically format your copy of the code with `ruff format`. Again remember to commit any automatically formatted files.

## Pre-commit hooks
We use the [pre-commit](pre-commit.com) framework for code and docstring styling
validation. It is highly recommended that you install `pre-commit` (included in 
the `dev` group) with `pre-commit install` so that the hooks will run and update
changes on every commit.

The hooks are run as part of our CI, so you're not required to use pre-commit, but
this may lead to failing tests on CI.


## Pytest unit testing

You can install the `pytest` unit testing toolkit with `pip install pytest`. You can then run `pytest` in the same directory as the `pyproject.toml` file to run the existing unit tests. Any test failures will report detailed debugging information. Note that the tests on github are run with the python version specified in the lock file, and the latest PyPI releases of the relevant dependencies (h5py, unyt, etc.). To run only tests in a specific file, you can do e.g. `pytest tests/test_creation.py`. The tests to be run can be further narrowed down with the `-k` argument to `pytest` (see `pytest --help`).

Note that many of the tests are implemented via [Hypothesis](https://hypothesis.readthedocs.io), and are a bit more advanced than basic unit tests,

A pixi task has been created for running tests that will also generate the
coverage report: `pixi run test`. The standard pytest arguments can simply be
added to this command if desired.

## Docstrings

Ruff currently has limited support for [numpydoc](https://numpydoc.readthedocs.io)-style docstrings. To run additional checks on docstrings use `numpydoc` (not provided), like `numpydoc lint **/*.py` in the same directory as the `pyproject.toml` file. This should not be necessary, however.

## Documentation
> [!NOTE]
> The documentation building tools are in the `docs` dependency group
> and are **not** part of `dev`!

The API documentation is built automatically from the docstrings of classes, functions, etc. in the source files. These follow the NumPy-style format. All public (i.e. not starting in `_`) modules, functions, classes, methods, etc. should have an appropriate docstring. Tests should also have descriptive docstrings, but full descriptions (e.g. of all parameters) are not required.

In addition to this there is "narrative documentation" that should describe the 
features of the code. The docs are built with `zensical` and `mkdocstrings` and
use the "ReadTheDocs" theme. If you have the dependencies installed, you can build
the documentation locally with `zensical serve` (e.g. `pixi run zserve`). This will then open your browser, allowing you to browse the documentation
and check your contributions.

This contributing guide has been adapted from the 
[swiftsimio CONTRIBUTING guide](https://github.com/SWIFTSIM/swiftsimio/blob/master/CONTRIBUTING.md).