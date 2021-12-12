import os
import sys

from setuptools import setup, find_packages

os.chdir(os.path.dirname(sys.argv[0]) or ".")

setup(
    name='tmu',
    version='0.3.5',
    url='https://github.com/cair/tmu/',
    author='Ole-Christoffer Granmo',
    author_email='ole.granmo@uia.no',
    license='MIT',
    description='Implements the Tsetlin Machine, Coalesced Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features and drop clause. TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating.',
    long_description='Implements the Tsetlin Machine, Coalesced Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features and drop clause. TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating.',

    setup_requires=["cffi>=1.0.0"],
    packages=find_packages(),
    cffi_modules=["./tmu/clause_bank_extension_build.py:ffibuilder", "./tmu/tools_extension_build.py:ffibuilder", "./tmu/weight_bank_extension_build.py:ffibuilder", ],
    install_requires=["cffi>=1.0.0"],
)
