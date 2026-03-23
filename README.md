# ContinuousDPs.jl

[![Build Status](https://github.com/QuantEcon/ContinuousDPs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantEcon/ContinuousDPs.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuantEcon/ContinuousDPs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/QuantEcon/ContinuousDPs.jl)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://QuantEcon.github.io/ContinuousDPs.jl/dev/)

Routines for solving continuous state dynamic programs by the Bellman equation collocation method.

## Installation

To install the package, open the Julia package manager (Pkg) and type:

```
add https://github.com/QuantEcon/ContinuousDPs.jl
```

## Supported problem class


## Example usage
Consider the deterministic optimal growth case:

```Julia
using ContinuousDPs, BasisMatrices

alpha = 0.65
beta = 0.95

f(s, x) = log(x)
g(s, x, e) = s^alpha - x

shocks = [1.]
weights = [1.]
x_lb(s) = 0
x_ub(s) = s

s_init = 0.1
ts_length = 25

n = 30
s_min, s_max = 0.1, 2.
basis = Basis(ChebParams(n, s_min, s_max))

# Solve DP
cdp = ContinuousDP(f, g, beta, shocks, weights, x_lb, x_ub, basis);
res = solve(cdp, PFI, tol=sqrt(eps()), max_iter=500);

# Value and policy on collocation nodes
res.V
res.X

# Evaluate on grids
eval_nodes = collect(range(s_min, stop=s_max, length=500))
set_eval_nodes!(res, eval_nodes)

# Simulate the state process
s_path = simulate(res, s_init, ts_length)

```



## Demo Notebooks

* [Stochastic Optimal Growth Model](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_optgrowth_jl.ipynb)
* [Examples from Miranda and Fackler 2002, Chapter 9](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_MF_jl.ipynb)
* [LQ Approximation with `QuantEcon.jl` and `ContinuousDPs.jl`](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/lqapprox_jl.ipynb)


## References
