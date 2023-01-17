

try:
    import tmu.tmulib
except ImportError:
    raise ImportError("Could not import cffi compiled libraries. To fix this problem, run pip install -e .")
