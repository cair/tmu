import logging.config
import pathlib
import json


if not logging.getLogger().handlers:
    # TODO add argparse configs here.
    _current_file = pathlib.Path(__file__).parent
    with _current_file.joinpath("logging_example.json").open("r") as config_file:
        logging.config.dictConfig(json.load(config_file))

try:
    import tmu.tmulib
except ImportError:
    raise ImportError("Could not import cffi compiled libraries. To fix this problem, run pip install -e .")


__version__ = "0.7.9"
