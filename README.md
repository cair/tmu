# Tsetlin Machine Unified - One Codebase to Rule Them All
![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square) ![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)![Maintenance](https://img.shields.io/maintenance/yes/2023?style=flat-square)

The TMU repository is a collection of Tsetlin Machine implementations, namely:
* Tsetlin Machine (https://arxiv.org/abs/1804.01508)
* Coalesced Tsetlin Machine (https://arxiv.org/abs/2108.07594)
* Convolutional Tsetlin Machine (https://arxiv.org/abs/1905.09688)
* Regression Tsetlin Machine (https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165)
* Weighted Tsetlin Machine (https://ieeexplore.ieee.org/document/9316190)
* Autoencoder (https://arxiv.org/abs/2301.00709)
* Multi-task classifier (to be published)
* One-vs-one multi-class classifier (to be published)
* Relational Tsetlin Machine (under development, https://link.springer.com/article/10.1007/s10844-021-00682-5)

Further, we implement many TM features, including:
* Support for continuous features (https://arxiv.org/abs/1905.04199)
* Drop clause (https://arxiv.org/abs/2105.14506)
* Literal budget (https://arxiv.org/abs/2301.08190)
* Focused negative sampling (https://ieeexplore.ieee.org/document/9923859)
* Type III Feedback (to be published)
* Incremental clause evaluation (to be published)
* Sparse computation with absorbing actions (to be published)

TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating.

# Installation

## Installing on Windows
To install on windows, you will need the MSVC build tools, [found here](https://visualstudio.microsoft.com/visual-cpp-build-tools/
).  When prompted, select the `Workloads → Desktop development with C++` package, 
which is roughly 6-7GB of size, install it and you should be able to compile the cffi modules.

## Installing TMU
```bash
pip install git+https://github.com/cair/tmu.git
```
