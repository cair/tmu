import os
import sys

from setuptools import setup, find_packages

os.chdir(os.path.dirname(sys.argv[0]) or ".")

setup(
    name='tmu2',
    version='0.6.0',
    url='https://github.com/cair/tmu/',
    author='Ole-Christoffer Granmo',
    author_email='ole.granmo@uia.no',
    license='MIT',
    description='Implements the Tsetlin Machine, Coalesced Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features, drop clause, Type III Feedback, focused negative sampling, multi-task classifier, autoencoder, literal budget, and one-vs-one multi-class classifier. TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating.',
    long_description='Implements the Tsetlin Machine (https://arxiv.org/abs/1804.01508), Coalesced Tsetlin Machine (https://arxiv.org/abs/2108.07594), Convolutional Tsetlin Machine (https://arxiv.org/abs/1905.09688), Regression Tsetlin Machine (https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165), and Weighted Tsetlin Machine (https://ieeexplore.ieee.org/document/9316190), with support for continuous features (https://arxiv.org/abs/1905.04199), drop clause (https://arxiv.org/abs/2105.14506), Type III Feedback (to be published), focused negative sampling (to be published), multi-task classifier (to be published), autoencoder (to be published), literal budget (to be published), and one-vs-one multi-class classifier (to be published). TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating.',

    setup_requires=["cffi>=1.0.0"],
    packages=find_packages(),
    cffi_modules=["./tmu/clause_bank_extension_build.py:ffibuilder", "./tmu/tools_extension_build.py:ffibuilder", "./tmu/weight_bank_extension_build.py:ffibuilder", ],
    install_requires=["cffi>=1.0.0"],
)
