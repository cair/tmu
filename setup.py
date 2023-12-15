from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import cffi
import tomli
from typing import Dict, Any

from setuptools.dist import Distribution

project_dir = Path(__file__).parent


def load_configuration(file_path: Path) -> Dict[str, Any]:
    """Load and parse the configuration from a given TOML file."""
    return tomli.loads(file_path.read_text())


def build_cffi():
    """
    Build the CFFI modules as per the configuration in `pyproject.toml`.
    Reads the configuration from `tool.cffi_builder` section.
    """
    config = load_configuration(project_dir / "pyproject.toml")
    cffi_builder_config = config.get("tool", {}).get("cffi_builder", {})

    sources = [Path(s) for s in cffi_builder_config.get("sources", [])]
    headers = [Path(s) for s in cffi_builder_config.get("headers", [])]
    include_dir = cffi_builder_config.get("include_dir", ".")


    flags = cffi_builder_config.get("flags", [])

    source_content = '\n'.join(s.read_text() for s in sources)
    header_content = '\n'.join(h.read_text() for h in headers)
    ffibuilder = cffi.FFI()
    ffibuilder.cdef(header_content)
    ffibuilder.set_source(
        cffi_builder_config.get("module_name", "tmu.tmulib"),
        source_content,
        include_dirs=[Path(include_dir).absolute()],
        extra_compile_args=flags
    )
    ffibuilder.compile(verbose=True)
    return ffibuilder


setup(
    include_package_data=True,
    packages=find_packages(),
    ext_modules=[build_cffi().distutils_extension()]
)
