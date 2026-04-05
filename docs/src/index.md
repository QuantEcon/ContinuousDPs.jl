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

## Problem formulation and interface

`ContinuousDPs.jl` solves infinite-horizon dynamic programs of the form

```math
V(s)
= \max_{x\in[x_{\mathrm{lb}}(s), x_{\mathrm{ub}}(s)]}
    \left \{
        f(s,x) + \beta \mathbb{E}_{\varepsilon}
            \left [ V(g(s,x,\varepsilon)) \right ]
    \right \}
```
where
- ``s \in \mathbb{R}^N`` is the **state** (continuous, possibly multi-dimensional),
- ``x \in \mathbb{R}`` is the **action** (continuous, 1-dimensional),
- ``f(s, x)`` is the **reward** function,
- ``g(s, x, \varepsilon)`` is the **state transition** function,
- ``\varepsilon`` is a **random shock**,
    (i.i.d. across periods, independent of the state and the action),
- ``\beta \in (0, 1)`` is the **discount factor**, and
- ``x_{\mathrm{lb}}(s)`` and ``x_{\mathrm{ub}}(s)`` are state-dependent
    **action bounds**.

This package employs the **Bellman equation collocation method** (Miranda and
Fackler 2002, Chapter 9): The value function ``V`` is approximated by a linear
combination of basis functions (Chebyshev polynomials, B-splines, or linear
functions) and is required to satisfy the Bellman equation at the collocation
nodes.
The package builds on [`BasisMatrices.jl`](https://github.com/QuantEcon/BasisMatrices.jl)
for basis construction and interpolation.

To solve the problem, construct a `ContinuousDP` instance by passing the
primitives of the model:

```Julia
cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, basis)
```
where
- `f`, `g`, `x_lb`, and `x_ub` are callable objects that represent the reward
  function, the state transition function, and the lower and upper action
  bounds functions, respectively,
- `discount` is the discount factor,
- `shocks` and `weights` specify a discretization of the distribution of
  ``\varepsilon`` (a vector of nodes and their probability weights), and
- `basis` is a `Basis` object from
  [`BasisMatrices.jl`](https://github.com/QuantEcon/BasisMatrices.jl) that
  contains the interpolation basis information.

Then call `solve(cdp)` to obtain the value function, policy function, and
residuals.

## Example usage

Solve a stochastic optimal growth model:

```Julia
using BasisMatrices, ContinuousDPs, QuantEcon

# Model primitives
function OptimalGrowthModel(;
        alpha = 0.4, beta = 0.96, s_min = 1e-5, s_max = 4.,
        mu = 0.0, sigma = 0.1
    )
    f(s, x) = log(x)
    g(s, x, e) = (s - x)^alpha * e
    x_lb(s) = s_min
    x_ub(s) = s
    return (; alpha, beta, s_min, s_max, mu, sigma,
            f, g, x_lb, x_ub)
end

p = OptimalGrowthModel()

# Lognormal quadrature nodes and weights from QuantEcon.jl
shocks, weights = qnwlogn(7, p.mu, p.sigma^2)

# Chebyshev basis from BasisMatrices.jl
basis = Basis(ChebParams(30, p.s_min, p.s_max))

# Construct and solve the DP
cdp = ContinuousDP(p.f, p.g, p.beta, shocks, weights, p.x_lb, p.x_ub, basis);
res = solve(cdp);

# Set evaluation nodes to finer grid
grid_y = collect(range(p.s_min, stop=p.s_max, length=200))
set_eval_nodes!(res, grid_y);

res.V  # Value function on evaluation grid
res.X  # Policy function on evaluation grid
res.resid  # Bellman equation residual

# Simulate a sample path of the state variable
s_init = 0.1
ts_length = 100
simulate(res, s_init, ts_length)
```

See the demo notebooks for further examples.

## Demo Notebooks

* [Stochastic Optimal Growth Model](https://nbviewer.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_optgrowth_jl.ipynb)
* [Examples from Miranda and Fackler 2002, Chapter 9](https://nbviewer.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_MF_jl.ipynb)
* [LQ Approximation with `QuantEcon.jl` and `ContinuousDPs.jl`](https://nbviewer.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/lqapprox_jl.ipynb)
