import logging.config
import logging
import pathlib
import json

_LOGGER = logging.getLogger(__name__)

if not logging.getLogger().handlers:

    _current_file = pathlib.Path(__file__).parent
    _default_logging_file = _current_file.joinpath("logging_example.json")
    if _default_logging_file.exists():
        with _default_logging_file.open("r") as config_file:
            logging.config.dictConfig(json.load(config_file))
    else:
        _LOGGER.warning(f"Could not find default_logging file: {_default_logging_file.absolute()}. This is most "
                        f"likely a bug.")

try:
    import tmu.tmulib
except ImportError as e:
    raise ImportError("Could not import cffi compiled libraries. To fix this problem, run pip install -e .", e)

__version__ = "0.8.2"
