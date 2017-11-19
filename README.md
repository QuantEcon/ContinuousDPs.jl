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
