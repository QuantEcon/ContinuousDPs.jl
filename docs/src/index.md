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

`ContinuousDPs.jl` solves infinite-horizon dynamic programs of the form

$$
V(s)
= \max_{x\in[x_{lb}(s), x_{ub}(s)]} 
    \left \{
        f(s,x) + \beta \mathbb{E}_{\varepsilon} 
            \left [ V(g(s,x,\varepsilon)) \right ] 
    \right \}
$$
where
- $s \in \mathbb{R}^d$ is the **state** (continuous, possibly multi-dimensional),
- $x \in \mathbb{R}$ is the **action** (continuous, 1-dimensional),
- $f(s, x)$ is the **reward** function,
- $g(s, x, \varepsilon)$ is the **state transition** function,
- $\varepsilon$ is a **random shock**,
    (i.i.d. across periods, independent of the state and the action),
- $\beta \in (0, 1)$ is the **discount factor**, and
- $x_{\mathrm{lb}}(s)$ and $x_{\mathrm{ub}}(s)$ are state-dependent
    **action bounds**.

This package employs the **Bellman equation collocation method** (Miranda and 
Fackler 2002, Chapter 9): The value function $ V $ is approximated by a linear 
combination of basis functions (Chebyshev polynomials, B-splines, or linear 
functions) and is required to satisfy the Bellman equation at the collocation 
nodes.

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
    $ \varepsilon $ (a vector of nodes and their probability weights), and
- `basis` is a `Basis` object from 
    [`BasisMatrices.jl`](https://github.com/QuantEcon/BasisMatrices.jl) that 
    contains the interpolation basis information.

Then call `solve(cdp)` to obtain the value function, policy function, and 
residuals.

## Example usage

A stochastic optimal growth model from a 
[QuantEcon lecture](https://julia.quantecon.org/dynamic_programming/optgrowth.html):

```Julia
using BasisMatrices
using ContinuousDPs
using Random
using PythonPlot

# For reproducible results
seed = 42
rng = MersenneTwister(seed);

# Specify the parameters
function OptimalGrowthModel(;
        alpha = 0.4, beta = 0.96, s_min = 1e-5, s_max = 4.,
        mu = 0.0, sigma = 0.1
    )
    f(s, x) = log(x)
    g(s, x, e) = (s - x)^alpha * e
    x_lb(s) = s_min
    x_ub(s) = s
    return (; alpha, beta, s_min, s_max, mu, sigma,
            f, g, x_lb, x_ub)  # NamedTuple
end

p = OptimalGrowthModel();

# Set shocks and weights
shock_size = 250
shocks = exp.(p.mu .+ p.sigma * randn(rng, shock_size))
weights = fill(1/shock_size, shock_size);

# Construct a `Basis` object
n = 30
basis = Basis(ChebParams(n, p.s_min, p.s_max))

# Solve DP
cdp = ContinuousDP(p.f, p.g, p.beta, shocks, weights, p.x_lb, p.x_ub, basis);
res = solve(cdp);

# Evaluate on grids
grid_size = 200
grid_y = collect(range(p.s_min, stop=p.s_max, length=grid_size))
set_eval_nodes!(res, grid_y);
```

See the demo notebooks for further examples.

## Demo Notebooks

* [Stochastic Optimal Growth Model](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_optgrowth_jl.ipynb)
* [Examples from Miranda and Fackler 2002, Chapter 9](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_MF_jl.ipynb)
* [LQ Approximation with `QuantEcon.jl` and `ContinuousDPs.jl`](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/lqapprox_jl.ipynb)

