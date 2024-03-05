from skbuild import setup  # This leverages scikit-build for CMake projects
import sysconfig

site_packages_path = sysconfig.get_paths()["purelib"]
python_root = sysconfig.get_paths()["data"]

# get Relative path between python root and site-packages
relative_path = site_packages_path[len(python_root):]

print(f"site_packages_path: {site_packages_path}")
print(f"python_root: {python_root}")
print(f"relative_path: {relative_path}")
setup(
    name="tmulibpy",
    version="0.1.0",
    description="A Python wrapper for tmulibpp library",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/tumlibpp",
    license="MIT",
    cmake_args=[
        '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_PYTHON=ON',
        '-DBUILD_EXECUTABLE=OFF'
    ],
    cmake_install_dir=f".{relative_path}",
)
