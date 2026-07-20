# ContinuousDPs.jl

[![Build Status](https://github.com/QuantEcon/ContinuousDPs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantEcon/ContinuousDPs.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuantEcon/ContinuousDPs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/QuantEcon/ContinuousDPs.jl)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://QuantEcon.github.io/ContinuousDPs.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://QuantEcon.github.io/ContinuousDPs.jl/dev/)

Routines for solving continuous state dynamic programs by the Bellman equation collocation method.

## Installation

To install the package, open the Julia package manager (Pkg) and type:

```
add ContinuousDPs
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
- ``x \in \mathbb{R}`` is the **action** (continuous, 1-dimensional;
    multi-dimensional and discrete actions are also supported --- see below),
- ``f(s, x)`` is the **reward** function,
- ``g(s, x, \varepsilon)`` is the **state transition** function,
- ``\varepsilon`` is a **random shock**
    (i.i.d. across periods; state- and action-dependent distributions are
    also supported --- see below),
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

To solve the problem, first construct a `ContinuousDP` instance by passing
the primitives of the model:

```Julia
cdp = ContinuousDP(f=f, g=g, discount=discount, x_lb=x_lb, x_ub=x_ub,
                   shocks=shocks, weights=weights)
```
where
- `f`, `g`, `x_lb`, and `x_ub` are callable objects that represent the reward
  function, the state transition function, and the lower and upper action
  bounds functions, respectively,
- `discount` is the discount factor, and
- `shocks` and `weights` specify a discretization of the distribution of
  ``\varepsilon`` (a vector of nodes and their probability weights).

Instead of `x_lb` and `x_ub`, an action space object can be passed as
`actions`:
`ContinuousActions{M}(x_lb, x_ub)` for an `M`-dimensional box of continuous
actions (with the bound functions returning length-`M` tuples or
vectors; policy
functions are then stored as `n x M` matrices), or `DiscreteActions(vals)`
for a finite set of actions of arbitrary type (solved by exact enumeration,
with `res.X_ind` holding the indices of the optimal actions into `vals`). `DiscreteActions(vals)` represents a fixed finite action set; for
state-dependent infeasibility, return `-Inf` from `f(s, x)`.

### State- and action-dependent shock distributions

The distribution of the shock may depend on the current state and action.
Three patterns, in increasing order of generality:

1. **Distribution shifted by the state or action** (e.g. a productivity
   shock whose mean depends on ``s``): absorb the dependence into the
   transition function `g` itself, keeping fixed `shocks`/`weights` for a
   baseline innovation. No special support is needed.
2. **State-dependent probabilities over fixed outcomes** (e.g. a disaster
   that occurs with probability ``p(s)``): pass a *callable* as `weights`.
   A function `weights(s)` is called with the current state and must
   return the probability weights of the shock nodes at `s` (one weight
   per node).
3. **Action-dependent probabilities** (e.g. a search effort ``x`` that
   succeeds with probability ``\lambda(x)``): pass a two-argument callable
   `weights(s, x)`. Since the weights then enter the first-order condition,
   the inner maximization automatically uses the derivative-free Brent
   method in this case.

A callable `weights` returning a `Tuple` (or a statically-sized vector
such as a `StaticArrays.SVector`) keeps the solver's inner loops
allocation-free; returning a freshly allocated `Vector` also works at a
small cost. The weights are not validated beyond a length check: weights
that sum to less than one are permitted and act as additional discounting
(e.g. exogenous exit risk). As always, keep `g(s, x, e)` within the
approximation domain for every shock node with positive weight.

The solution methodology --- the interpolation basis and the algorithm
parameters --- is specified separately by a solver object:

```Julia
solver = CollocationSolver(basis; algorithm=PFI)  # or algorithm=VFI
res = solve(cdp, solver)
```
where `basis` is a `Basis` object from
[`BasisMatrices.jl`](https://github.com/QuantEcon/BasisMatrices.jl) that
contains the interpolation basis information; its domain is the
approximation domain of the value function. `solve` returns the value
function, policy function, and residuals. The inner maximization over
continuous actions is solved via the first-order condition by default; pass
`inner_solver=:brent` to `CollocationSolver` for a derivative-free method,
and `tol` and `max_iter` to control the iteration. For linear-quadratic
approximation around a reference point, pass `LQASolver(basis; point=(s, x, e))`
to `solve` instead.

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

# Construct the DP with the model primitives
cdp = ContinuousDP(f=p.f, g=p.g, discount=p.beta, x_lb=p.x_lb, x_ub=p.x_ub,
                   shocks=shocks, weights=weights);

# Solve by collocation with a Chebyshev basis from BasisMatrices.jl
basis = Basis(ChebParams(30, p.s_min, p.s_max))
res = solve(cdp, CollocationSolver(basis));

# Set evaluation nodes to finer grid
grid_y = collect(range(p.s_min, stop=p.s_max, length=200))
set_eval_nodes!(res, grid_y);

res.V  # Value function on evaluation grid
res.X  # Policy function on evaluation grid
res.resid  # Bellman equation residuals on evaluation grid

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
