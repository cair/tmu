import os
import glob

__all__ = []
directory = os.path.dirname(__file__)

for filename in glob.glob(os.path.join(directory, '*.py')):
    module = os.path.basename(filename)[:-3]  # Remove .py extension
    if module != '__init__':
        __import__(__name__ + '.' + module)
        __all__.append(module)