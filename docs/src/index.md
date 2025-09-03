# ContinuousDPs.jl

[![Build Status](https://github.com/QuantEcon/ContinuousDPs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantEcon/ContinuousDPs.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuantEcon/ContinuousDPs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/QuantEcon/ContinuousDPs.jl)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://QuantEcon.github.io/ContinuousDPs.jl/dev/)

Routines for solving continuous state dynamic programs by the Bellman equation collocation method.

## Installation

To install the package, open the Julia package manager (Pkg) and type:

```julia
add https://github.com/QuantEcon/ContinuousDPs.jl
```

## Quick Start

```julia
using ContinuousDPs

# Create a continuous dynamic programming problem
# (example code here would depend on the specific API)
```

## Demo Notebooks

* [Stochastic optimal growth model](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_optgrowth_jl.ipynb)
* [Examples from Miranda and Fackler 2002, Chapter 9](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_MF_jl.ipynb)

## References

* M. J. Miranda and P. L. Fackler, Applied Computational Economics and Finance, MIT press, 2002.