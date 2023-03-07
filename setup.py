import codecs
import os
from pathlib import Path
from setuptools import setup, find_packages

current_dir = Path(__file__).parent

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='tmu',
    version=get_version("tmu/__init__.py"),  # Change version in tmu/__init__.py
    url='https://github.com/cair/tmu/',
    author='Ole-Christoffer Granmo',
    author_email='ole.granmo@uia.no',
    license='MIT',
    description='Implements the Tsetlin Machine, Coalesced Tsetlin Machine, Convolutional Tsetlin Machine, Regression '
                'Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features, drop clause, '
                'Type III Feedback, focused negative sampling, multi-task classifier, autoencoder, literal budget,'
                'incremental clause evaluation, sparse computation with absorbing exclude state, and one-vs-one multi-class classifier. TMU is written in Python with '
                'wrappers for C and CUDA-based clause evaluation and updating.',
    long_description='Implements the Tsetlin Machine (https://arxiv.org/abs/1804.01508), Coalesced Tsetlin Machine ('
                     'https://arxiv.org/abs/2108.07594), Convolutional Tsetlin Machine ('
                     'https://arxiv.org/abs/1905.09688), Regression Tsetlin Machine ('
                     'https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165), and Weighted Tsetlin '
                     'Machine (https://ieeexplore.ieee.org/document/9316190), with support for continuous features ('
                     'https://arxiv.org/abs/1905.04199), drop clause (https://arxiv.org/abs/2105.14506), '
                     'Type III Feedback (to be published), focused negative sampling ('
                     'https://ieeexplore.ieee.org/document/9923859), multi-task classifier (to be published), '
                     'autoencoder (https://arxiv.org/abs/2301.00709), literal budget (https://arxiv.org/abs/2301.08190), incremental '
                     'clause evaluation (to be published), sparse computation with absorbing exclude state (to be published), and one-vs-one multi-class classifier (to be published). '
                     'TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating.',
    packages=find_packages(),
    cffi_modules=[
        "tmu/lib/tmulib_extension_build.py:ffibuilder"
    ],
    install_requires=[
        "cffi>=1.0.0",
        "numpy",
        "pandas",
        "scikit-learn"
    ]
)
