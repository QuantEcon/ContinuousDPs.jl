# ContinuousDPs.jl
Routines for solving continuous state dynamic programs by the Bellman equation collocation method

## Installation

This package at the moment depends on the `master` version of `BasisMatrices.jl`:

```jl
Pkg.add("BasisMatrices");
Pkg.checkout("BasisMatrices", "master")
```

Then install `ContinuousDPs.jl` by

```jl
Pkg.clone("https://github.com/QuantEcon/ContinuousDPs.jl")
```

## Demo Notebooks

* [Stochastic optimal growth model](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/master/examples/cdp_ex_optgrowth_jl.ipynb)
* [Examples from Miranda and Fackler 2002, Chapter 9](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/master/examples/cdp_ex_MF_jl.ipynb)
