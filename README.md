# ddplib

A library of functions for the disturbance decoupling problem (DDP) of linear
systems, based on the geometric approach. It is still under development.

## Installation

```
pip install .
```

This also installs the dependencies: NumPy, SciPy and the
[Python Control Systems Library](https://python-control.readthedocs.io/).

## Usage

For the system `xdot = Ax + Bu + Ed`, `y = Hx`:

```python
import numpy as np
from ddplib import ddp

A = np.array([[0., 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
B = np.array([[0., 0], [1, 0], [0, 0], [1, 1]])
H = np.array([[1., 0, 0, 0]])
E = np.array([[0.], [0], [1], [0]])

solvable, V = ddp.is_ddp_solvable(A, B, H, E)   # V: maximal controlled invariant subspace in ker H
R = ddp.controllability_subspace(A, B, H)        # supremal controllability subspace in ker H
F = ddp.ddp_place(A, B, R, [-1, -2, -3, -4])     # eig(A + BF) = roots and (A + BF) R is in R
```

Subspaces are passed and returned as matrices whose columns span them; the zero
subspace is returned as a single zero column. Rank decisions use the relative
tolerance `ddp.rtol` (default `1e-10`).

## Tests

```
pip install -e '.[test]'
pytest
```

## References

1. W. M. Wonham, *Linear Multivariable Control: A Geometric Approach*.
2. H. L. Trentelman, A. A. Stoorvogel, M. Hautus, *Control Theory for Linear Systems*.
