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
        f(s,x) + \beta \mathbb{E}_{\varepsilon} \left [ V(g(s,x,\varepsilon)) \right ] 
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
- $x_{\mathrm{lb}}(s)$ and $x_{\mathrm{ub}}(s)$ are state-dependent **action bounds**.

This package employs the **Bellman equation collocation method** (Miranda and Fackler 2002, Chapter 9): The value function $ V $
 is approximated by a linear combination of basis functions (Chebyshev polynomials, B-splines, or linear functions) and is required to satisfy the Bellman equation at the collocation nodes.

To solve the problem, construct a `ContinuousDP` instance by passing the primitives of the model:
```Julia
cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, basis)
```
where
- `f`, `g`, `x_lb`, and `x_ub` are callable objects that represent the reward function, the state transition function, and the lower and upper action bounds functions, respectively,
- `discount` is the discount factor,
- `shocks` and `weights` specify a discretization of the distribution of $ \varepsilon $ (a vector of nodes and their probability weights), and
- `basis` is a `Basis` object from [`BasisMatrices.jl`](https://github.com/QuantEcon/BasisMatrices.jl) that contains the interpolation basis information.

Then call `solve(cdp)` to obtain the value function, policy function, and residuals.

## Example usage

A deterministic optimal growth case:

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

See the demo notebooks for further examples.

## Demo Notebooks

* [Stochastic Optimal Growth Model](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_optgrowth_jl.ipynb)
* [Examples from Miranda and Fackler 2002, Chapter 9](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_MF_jl.ipynb)
* [LQ Approximation with `QuantEcon.jl` and `ContinuousDPs.jl`](http://nbviewer.jupyter.org/github/QuantEcon/ContinuousDPs.jl/blob/main/examples/lqapprox_jl.ipynb)

