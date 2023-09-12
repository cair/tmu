# Tsetlin Machine Unified (TMU) - One Codebase to Rule Them All
![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square) ![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)![Maintenance](https://img.shields.io/maintenance/yes/2023?style=flat-square)

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
    - Type III Feedback *(Upcoming)*
    - Incremental Clause Evaluation *(Upcoming)*
    - Sparse Computation with Absorbing Actions *(Upcoming)*
    - TMComposites: Plug-and-Play Collaboration Between Specialized Tsetlin Machines *([In Progress](https://arxiv.org/abs/2309.04801))*

- Wrappers for C and CUDA-based clause evaluation and updates to enable high-performance computation.

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
pip install git+https://github.com/cair/tmu.git
```

## 🛠 Development

If you're looking to contribute or experiment with the codebase, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:cair/tmu.git
   ```

2. **Set Up Development Environment**:
   Navigate to the project directory and compile the C library:
   ```bash
   cd tmu && pip install develop .
   ```

3. **Starting a New Project**:
   For your projects, simply create a new folder within 'examples' and initiate your development.

#### Modifying the C Codebase
If you make changes to the C codebase, ensure you recompile the code using:
```bash
pip install develop .
```

---
