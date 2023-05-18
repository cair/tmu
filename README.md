# Tsetlin Machine Unified - One Codebase to Rule Them All
![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square) ![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)![Maintenance](https://img.shields.io/maintenance/yes/2023?style=flat-square)

The Tsetlin Machine Unified (TMU) repository is a central hub for several Tsetlin Machine implementations. It includes:
* [Tsetlin Machine](https://arxiv.org/abs/1804.01508)
* [Coalesced Tsetlin Machine](https://arxiv.org/abs/2108.07594)
* [Convolutional Tsetlin Machine](https://arxiv.org/abs/1905.09688)
* [Regression Tsetlin Machine](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165)
* [Weighted Tsetlin Machine](https://ieeexplore.ieee.org/document/9316190)
* [Autoencoder](https://arxiv.org/abs/2301.00709)
* Multi-task classifier (Forthcoming)
* One-vs-one multi-class classifier (Forthcoming)
* [Relational Tsetlin Machine](https://link.springer.com/article/10.1007/s10844-021-00682-5) (In Progress)

In addition to these, TMU also implements various Tsetlin Machine features, including:
* [Support for continuous features](https://arxiv.org/abs/1905.04199)
* [Drop clause](https://arxiv.org/abs/2105.14506)
* [Literal budget](https://arxiv.org/abs/2301.08190)
* [Focused negative sampling](https://ieeexplore.ieee.org/document/9923859)
* Type III Feedback (Forthcoming)
* Incremental clause evaluation (Forthcoming)
* Sparse computation with absorbing actions (Forthcoming)

TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating.

## Installation

### Requirements for Windows Installation
Windows installation requires MSVC build tools, which can be [downloaded here](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, select the `Workloads → Desktop development with C++` package. This package is approximately 6-7GB in size. Once installed, the cffi modules should compile successfully.

### TMU Installation
Use the following command to install TMU:

```bash
pip install git+https://github.com/cair/tmu.git
```

## Development
For development, we suggest cloning the repository and working directly on the codebase. Please ensure that you have added SSH keys to GitHub:

1. Clone the repository:

```bash
git clone git@github.com:cair/tmu.git
```

2. Compile C library:

```bash
cd tmu && pip install -e .
```

For specific projects, create a folder in 'examples' and begin your development.

### Modifying the C Codebase?
If you modify the C codebase, you must recompile using the `pip install -e .` command.

