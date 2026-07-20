# POMDPs.jl Interface

ContinuousDPs.jl ships a package extension connecting the collocation
solver to the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl)
ecosystem. It is activated by loading both trigger packages:

```julia
using POMDPs, POMDPTools
```

The extension makes `CollocationSolver` available as a solver for
POMDPs.jl models: `POMDPs.solve(CollocationSolver(basis), m)` solves any
*explicit-finite* MDP --- continuous states covered by the basis,
finitely many actions, and explicit transition distributions --- by the
Bellman equation collocation method, and returns a standard
`POMDPs.Policy`. This gives POMDPs.jl models access to smooth global
approximation of the value function and the policy over a continuous
state space, complementing the ecosystem's sampling-based planners.

## Solving a POMDPs.jl model

A stochastic optimal growth model with the savings rate chosen from a
finite grid, written as a
[QuickPOMDPs](https://github.com/JuliaPOMDP/QuickPOMDPs.jl) model and
solved by collocation (for log utility and Cobb--Douglas technology the
analytic optimal savings rate is the constant `alpha * beta`):

```julia
using ContinuousDPs
using BasisMatrices: Basis, ChebParams
using QuantEcon: qnwlogn
using POMDPs, POMDPTools, QuickPOMDPs

function build_growth_mdp(; alpha = 0.4, beta = 0.95,
                          s_min = 0.1, s_max = 2.0)
    f(s, x) = log((1 - x) * s)                     # consume the rest
    g(s, x, e) = clamp((x * s)^alpha * e, s_min, s_max)
    shocks, weights = qnwlogn(7, 0.0, 0.05^2)
    x_grid = collect(0.05:0.05:0.95)               # savings rates
    return QuickMDP(
        statetype = Float64,
        actions = x_grid,
        discount = beta,
        transition = (s, x) -> SparseCat(g.(s, x, shocks), weights),
        reward = f,
    )
end

m = build_growth_mdp()

basis = Basis(ChebParams(30, 0.1, 2.0))
policy = POMDPs.solve(CollocationSolver(basis), m)

action(policy, 1.0)   # greedy savings rate at s = 1.0: 0.4,
                      # the grid point closest to alpha * beta = 0.38
value(policy, 1.0)    # fitted value function at s = 1.0

# The ecosystem's tooling works as usual, e.g. rollout simulation
POMDPs.simulate(RolloutSimulator(max_steps=100), m, policy, 1.0)
```

!!! note "Wrap model definitions in a build function"
    The build-function pattern above is not cosmetic. `QuickMDP`'s
    keyword functions are called inside the solver's innermost loops, and
    a top-level closure over a non-`const` global looks the global up on
    every call, degrading type inference and performance. Capture locals
    in a build function (as here), or make the captured objects `const`.

## Requirements and scope

`POMDPs.solve` checks its requirements at solve time and throws an
informative error when one fails:

- **Finite explicit actions.** `actions(m)` must return a finite
  collection. A state-dependent restriction via `actions(m, s)` is
  supported and mapped to infeasibility; every collocation node needs at
  least one feasible action. The model's `transition` and `reward` are
  never evaluated on infeasible state--action pairs.
- **Explicit transition distributions.** `transition(m, s, x)` must
  return a distribution with explicitly enumerable support and
  probabilities (`SparseCat`, `Deterministic`, ... --- anything
  supporting `POMDPTools.weighted_iterator`). Generative-only models are
  out of scope.
- **No terminal states** at any collocation node (not supported in this
  version).
- **Rewards** may be defined as `reward(m, s, x)` or
  `reward(m, s, x, sp)`. The arity is chosen once at solve time by a
  probe call, preferring the direct form; with only the
  next-state-dependent form, the expectation over the transition's
  branches is taken, at the cost of one reward call per branch per
  evaluation.
- **The state space is continuous**, with the domain and dimension given
  by the solver's basis. States are passed to the model as indexable
  coordinate points (a `Float64` for a one-dimensional basis, otherwise
  an indexable collection of coordinates), and the model's next states
  must be indexable likewise (scalars, tuples, or static vectors).
- **Next states must stay within the basis domain** for every feasible
  action, as everywhere in ContinuousDPs: the fitted value function is
  extrapolated outside the domain, which can silently destroy a solve
  (astronomically so for Chebyshev bases). Clamp inside the transition,
  or choose the domain to be invariant.

## The returned policy

The returned `CollocationPolicy` is a standard `POMDPs.Policy`:

- `action(policy, s)` evaluates the computed policy by exact greedy
  recomputation at `s` (the solution's coefficients are interpolated;
  the discrete policy itself is never interpolated);
- `value(policy, s)` evaluates the fitted value function;
- the full `CDPSolveResult` remains available as `policy.res`
  (residuals, `set_eval_nodes!`, the native `simulate`).

`CollocationPolicy` is not thread-safe: for parallel rollouts, use one
policy instance per thread (the underlying evaluation caches are
single-threaded).

## Example: a belief MDP

Finite-observation learning problems reduce to belief MDPs whose
transitions are exactly the extension's explicit-finite shape. The job
search model with learning about the offer distribution
([examples/cdp_ex_odu.jl](https://github.com/QuantEcon/ContinuousDPs.jl/blob/main/examples/cdp_ex_odu.jl),
following [the QuantEcon lecture](https://julia.quantecon.org/dynamic_programming/odu.html))
solves natively as a `ContinuousDP` with state-dependent callable
`weights`; the same belief MDP written as a POMDPs.jl model solves
through the extension:

```julia
function build_belief_mdp(sp)   # sp = SearchProblem() from the example
    return QuickMDP(
        statetype = NTuple{2,Float64},
        actions = [:reject, :accept],
        discount = sp.beta,
        reward = (s, x) -> x === :accept ? s[1] : sp.c,
        transition = function (s, x)
            x === :accept && return Deterministic((s[1], s[2]))
            SparseCat([(wp, sp.update(wp, s[2])) for wp in sp.shocks],
                      sp.weights(s))
        end,
    )
end

bm = build_belief_mdp(sp)
basis = Basis(
    SplineParams(collect(range(0.0, sp.w_max, length=40)), 0, 3),
    SplineParams(collect(range(sp.pi_min, sp.pi_max, length=40)), 0, 3))
policy = POMDPs.solve(CollocationSolver(basis), bm)

POMDPs.action(policy, (1.5, 0.5))
hist = POMDPs.simulate(HistoryRecorder(max_steps=30), bm, policy, (1.0, 0.5))
```

The two formulations are the same transition kernel in two vocabularies:
the `SparseCat` pairing each offer with its Bayes-updated belief and the
belief-mixed probabilities `sp.weights(s)` is precisely the native
model's `g` over `shocks` weighted by the callable `weights`. The native
and POMDPs solves produce identical coefficients.

## Two `solve` generic functions

| Call | Generic function | Returns |
|---|---|---|
| `solve(cdp::ContinuousDP, solver)` | `QuantEcon.solve` (extended by ContinuousDPs) | `CDPSolveResult` |
| `solve(solver::CollocationSolver, m)` | `POMDPs.solve` | `CollocationPolicy` |

QuantEcon.jl and POMDPs.jl both export `solve` *and* `simulate`, so
under `using QuantEcon, POMDPs` (with or without ContinuousDPs) the bare
names are ambiguous and qualified calls (`POMDPs.solve`,
`POMDPs.simulate`) are needed, as in the examples above. This name clash
between the two ecosystems predates this extension.

Note that `CollocationSolver` does not subtype `POMDPs.Solver`:
`POMDPs.solve` is an empty generic function, so the methods above work
by ordinary dispatch. Subtyping would require an unconditional POMDPs
dependency in the core package; if you need a `POMDPs.Solver`-typed
object (e.g. for a solver-annotated container), wrap the solver in a
small struct of your own.
