Implements the Tsetlin Machine
==================================

- `Tsetlin Machine <https://arxiv.org/abs/1804.01508>`_
- `Coalesced Tsetlin Machine <https://arxiv.org/abs/2108.07594>`_
- `Convolutional Tsetlin Machine <https://arxiv.org/abs/1905.09688>`_
- `Regression Tsetlin Machine <https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165>`_
- `Weighted Tsetlin Machine <https://ieeexplore.ieee.org/document/9316190>`_

Features and Extensions
=======================

- Support for continuous features: `<https://arxiv.org/abs/1905.04199>`_
- Drop clause: `<https://arxiv.org/abs/2105.14506>`_
- Type III Feedback (to be published)
- Focused negative sampling: `<https://ieeexplore.ieee.org/document/9923859>`_
- Multi-task classifier (to be published)
- Autoencoder: `<https://arxiv.org/abs/2301.00709>`_
- Literal budget: `<https://arxiv.org/abs/2301.08190>`_
- Incremental clause evaluation (to be published)
- Sparse computation with absorbing exclude (to be published)
- One-vs-one multi-class classifier (to be published)

Technical Details
=================

TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating.
