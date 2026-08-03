# Tsetlin Machine Unified (TMU) - One Codebase to Rule Them All
![License](https://img.shields.io/github/license/cair/tmu.svg?style=flat-square) ![Python Version](https://img.shields.io/pypi/pyversions/tmu.svg?style=flat-square) ![Maintenance](https://img.shields.io/maintenance/yes/2026?style=flat-square)

TMU is a comprehensive repository that encompasses several Tsetlin Machine implementations. Offering a rich set of features and extensions, it serves as a central resource for enthusiasts and researchers alike.

## Features
- Core Implementations:
    - [Tsetlin Machine](https://arxiv.org/abs/1804.01508)
    - [Coalesced Tsetlin Machine](https://arxiv.org/abs/2108.07594)
    - [Convolutional Tsetlin Machine](https://arxiv.org/abs/1905.09688)
    - [Regression Tsetlin Machine](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165)
    - [Weighted Tsetlin Machine](https://ieeexplore.ieee.org/document/9316190)
    - [Autoencoder](https://arxiv.org/abs/2301.00709)
    - Multi-task Classifier *(Upcoming)*
    - One-vs-one Multi-class Classifier *(Upcoming)*
    - [Relational Tsetlin Machine](https://link.springer.com/article/10.1007/s10844-021-00682-5) *(In Progress)*

- Extended Features:
    - [Support for Continuous Features](https://arxiv.org/abs/1905.04199)
    - [Drop Clause](https://arxiv.org/abs/2105.14506)
    - [Literal Budget](https://arxiv.org/abs/2301.08190)
    - [Focused Negative Sampling](https://ieeexplore.ieee.org/document/9923859)
    - [Type III Feedback](https://arxiv.org/abs/2309.06315)
    - Incremental Clause Evaluation *(Upcoming)*
    - [Sparse Computation with Absorbing Actions](https://arxiv.org/abs/2310.11481)
    - TMComposites: Plug-and-Play Collaboration Between Specialized Tsetlin Machines *([In Progress](https://arxiv.org/abs/2309.04801))*

- Wrappers for C and CUDA-based clause evaluation and updates to enable high-performance computation.

## Guides and Tutorials
- [Setting up efficient Development Environment](docs/tutorials/devcontainers/devcontainers.md)

## 📦 Installation

#### **Prerequisites for Windows**
Before installing TMU on Windows, ensure you have the MSVC build tools. Follow these steps:
1. [Download MSVC build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install the `Workloads → Desktop development with C++` package. *(Note: The package size is about 6-7GB.)*

#### **Dependencies**
Ubuntu: `sudo apt install libffi-dev`

#### **Installing TMU**
To get started with TMU, run the following command:
```bash
# Installing Stable Branch
pip install git+https://github.com/cair/tmu.git

# Installing Development Branch
pip install git+https://github.com/cair/tmu.git@dev
```

## 🛠 Development

If you're looking to contribute or experiment with the codebase, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone -b dev git@github.com:cair/tmu.git && cd tmu
   ```

2. **Set Up Development Environment**:
   Navigate to the project directory and compile the C library:
   ```bash
   # Install TMU
    pip install .
   
   # (Alternative): Install TMU in Development Mode
    pip install -e .
   
   # Install TMU-Composite
    pip install .[composite]
   
   # Install TMU-Composite in Development Mode
    pip install -e .[composite]
   ```

3. **Starting a New Project**:
   For your projects, simply create a new **branch** and then within the 'examples' folder, create a new project and initiate your development.

---
