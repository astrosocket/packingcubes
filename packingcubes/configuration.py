import logging
import os
import tomllib
from array import array
from pathlib import Path
from warnings import warn

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


def _determine_field_format():
    formats = ["H", "I", "L", "Q"]
    for f in formats:
        if array(f).itemsize == 4:
            return f
    if array(formats[0]).itemsize > 4:
        return formats[0]
    raise NotImplementedError("None of the included array formats are long enough!")


FIELD_FORMAT = _determine_field_format()


def _config_dir() -> str:
    config_root = os.environ.get(
        "XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")
    )
    return os.path.join(config_root, "astrosocket")


def _get_global_config_file() -> str:
    return os.path.join(_config_dir(), "packingcubes.toml")


def _get_local_config_file() -> str:
    path = Path.cwd()
    while path.parent is not path:
        candidate = path.joinpath("packingcubes.toml")
        if candidate.is_file():
            return os.path.abspath(candidate)
        path = path.parent

    return os.path.join(os.path.abspath(os.curdir), "packingcubes.toml")


def _get_default_cfg() -> dict:
    return {"test_data_dir": "/does/not/exist", "config_file": "(default)"}


def update_cfg(pccfg: dict, config_file: str):
    try:
        with open(config_file, "rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        warn(
            f"Could not load configuration file {config_file} (invalid TOML: {exc})",
            stacklevel=2,
        )
    else:
        LOGGER.info(f"Updating configuration from {config_file}")
        pccfg.update(data)
        pccfg["config_file"] = config_file


def get_packingcubes_config() -> dict:
    pccfg = _get_default_cfg()
    local_config = _get_local_config_file()
    global_config = _get_global_config_file()
    if os.path.exists(local_config):
        update_cfg(pccfg, local_config)
    elif os.path.exists(global_config):
        update_cfg(pccfg, global_config)

    return pccfg


pccfg = get_packingcubes_config()


def get_test_data_dir_path():
    p = Path(pccfg["test_data_dir"]).expanduser()
    if p.is_dir():
        return p
    warn(
        "Storage directory from packingcubes config doesn't exist"
        f"(currently set to '{p}'). "
        "Current working directory will be used instead.",
        stacklevel=2,
    )
    return Path.cwd()
