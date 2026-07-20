# ContinuousDPs.jl

ContinuousDPs.jl is a Julia package that provides routines for solving continuous state dynamic programs using the Bellman equation collocation method. It is part of the QuantEcon ecosystem and offers Policy Function Iteration (PFI) and Value Function Iteration (VFI) algorithms for solving dynamic programming problems.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup
- Julia 1.10+ is required. Check version: `julia --version`
- Install package dependencies: `julia --project=. -e "using Pkg; Pkg.instantiate()"` -- takes 75 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
- The package uses standard Julia package structure with `Project.toml` for dependencies

### Building and Testing
- Build the package: `julia --project=. -e "using Pkg; Pkg.build()"`
- Run all tests: `julia --project=. -e "using Pkg; Pkg.test()"` -- takes 60-120 seconds. NEVER CANCEL. Set timeout to 240+ seconds.
- Run basic functionality test: Create a simple script importing `ContinuousDPs`, `QuantEcon`, and `BasisMatrices` to verify the package works

### Development Workflow
- Start Julia REPL in package mode: `julia --project=.`
- Load the package in development: `julia --project=. -e "using ContinuousDPs"`
- Import required dependencies: `using ContinuousDPs; using BasisMatrices: Basis, ChebParams, SplineParams`
- `solve`, `VFI`, `PFI`, and `LQA` are exported by ContinuousDPs

## Core Functionality Testing

### Creating and Solving a Continuous DP
Always test new code with this basic workflow:
```julia
using ContinuousDPs
using BasisMatrices: Basis, ChebParams

# Approximation domain
s_min, s_max = 0.1, 2.0

# Define reward and transition functions
f(s, x) = log(x)  # reward function
# IMPORTANT: keep the next state within the approximation domain (see
# "Out-of-domain next states" below)
g(s, x, e) = clamp(s^0.3 * x^0.7 + e, s_min, s_max)  # state transition

# Setup problem parameters
discount = 0.9
shocks = [0.0]  # shock values
weights = [1.0]  # shock weights
x_lb(s) = 0.01  # lower bound on action
x_ub(s) = s - 0.01  # upper bound on action

# Create continuous DP from the model primitives (no basis here)
cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub)

# The solution methodology is specified by a solver object
n = 10
basis = Basis(ChebParams(n, s_min, s_max))
solver = CollocationSolver(basis; algorithm=PFI)  # or algorithm=VFI
res = solve(cdp, solver)

# Simulate solution
s_init = 1.0
ts_length = 10
s_path = simulate(res, s_init, ts_length)
```

### Validation Scenarios
ALWAYS run these validation steps after making changes:
1. **Package Loading Test**: Verify `using ContinuousDPs` works without errors
2. **Basic Solve Test**: Create a simple DP problem and solve it using both PFI and VFI
3. **Simulation Test**: Simulate paths from solved problems to ensure results are reasonable
4. **Integration Test**: Run the full test suite to ensure no regressions

## Repository Structure

### Key Directories and Files
```
├── src/
│   ├── ContinuousDPs.jl     # Main module file (include order = narrative order)
│   ├── point_eval.jl        # Non-allocating point-evaluation kernels
│   ├── cdp.jl               # Narrative core: types, operators, solve, simulate
│   ├── inner_solvers.jl     # Per-state inner maximization paths and sweeps
│   ├── policy_system.jl     # Policy-evaluation collocation system assembly
│   └── lq_approx.jl         # Linear quadratic approximation methods
├── docs/
│   ├── make.jl              # Documenter build script
│   └── src/                 # Documentation source files
├── test/
│   ├── runtests.jl              # Test runner
│   ├── test_point_eval.jl       # Agreement tests for the point kernels
│   ├── test_workspace.jl        # CDPWorkspace and in-place evaluation tests
│   ├── test_foc.jl              # First-order-condition inner solver tests
│   ├── test_evaluate_policy.jl  # Policy-evaluation assembly tests
│   ├── test_cdp.jl              # Tests for core CDP functionality
│   ├── test_cdp_multidim.jl     # Tests for multi-dimensional CDP
│   └── test_lq_approx.jl        # Tests for LQ approximation
├── benchmark/
│   ├── benchmarks.jl        # PkgBenchmark/BenchmarkTools suite (SUITE)
│   ├── Project.toml         # Benchmark environment
│   └── README.md            # How to run and compare benchmarks
├── examples/
│   ├── cdp_ex_optgrowth_jl.ipynb    # Optimal growth model example
│   ├── cdp_ex_MF_jl.ipynb           # Miranda & Fackler examples
│   └── lqapprox_jl.ipynb            # LQ approximation examples
└── Project.toml             # Package dependencies and metadata
```

### Important Files to Check When Making Changes
- Always check `src/cdp.jl` when modifying core solving algorithms; it is the narrative core (types, operators, solve, simulate) and must stay readable top-down — implementation detail belongs in `src/inner_solvers.jl` (per-state maximization) and `src/policy_system.jl` (system assembly), which are included after it and may reference its types (never the reverse)
- Always check `src/point_eval.jl` when modifying interpolant evaluation; its behavior contract is exact agreement with `BasisMatrices.funeval`/`evalbase` (see below)
- Always check `src/lq_approx.jl` when working with linear quadratic approximations
- Always run tests in `test/test_point_eval.jl` when modifying `src/point_eval.jl`
- Always run tests in `test/test_cdp.jl`, `test/test_cdp_multidim.jl`, `test/test_foc.jl`, and `test/test_evaluate_policy.jl` when modifying CDP functionality; add `test/test_transition_kernel.jl` when touching the transition kernel or the weights contract
- Always run tests in `test/test_lq_approx.jl` when modifying LQ approximation
- Update `benchmark/benchmarks.jl` when internal signatures used there change
- `README.md`, `docs/src/index.md`, and `examples/cdp_ex_optgrowth_jl.ipynb` share the problem-formulation and interface description and are kept synchronized content-wise: when editing one of them, update the other two to a contextually appropriate extent (they need not be verbatim copies; judge the extent by what changed)

## Architecture Notes (performance-critical internals)

### Point-evaluation kernels (`src/point_eval.jl`)
- `FunEvalCache(basis)` + `funeval_point!(fec, C, x)`: non-allocating single-point evaluation of an interpolant; `DerivFunEvalCache(basis, order)` + `set_coefs!` for partial derivatives via coefficient differentiation.
- CONTRACT: these kernels must reproduce `BasisMatrices.evalbase`/`funeval` semantics exactly, including behavior outside the interpolation domain; `test/test_point_eval.jl` enforces agreement at machine precision. Any change here requires those tests.
- The file is deliberately self-contained (no DP-specific types): it is planned to move upstream to BasisMatrices.jl (issue #94). Do not add ContinuousDPs types to it, and do not make other files depend on its imports (import what you use explicitly).

### Action spaces (`src/cdp.jl`)
- `ContinuousDP` stores an `actions::ActionSpace`: `ContinuousActions(x_lb, x_ub)` (one-dimensional box; the legacy positional `x_lb, x_ub` constructor arguments wrap into this), `ContinuousActions{M}(x_lb, x_ub)` (M-dimensional box with length-M tuple- or vector-valued bounds; actions passed to `f`/`g` as length-M indexable collections; policies stored as `n x M` matrices), or `DiscreteActions(vals)` (finite set of action values of any homogeneous type).
- Discrete actions follow the QuantEcon `MarkovChain`/`state_values` convention: solvers work with indices internally (`ws.X_ind`, `res.X_ind`); `res.X` exposes the corresponding values. The inner problem is solved by exact enumeration; `inner_solver` is ignored. Infeasible state-action pairs are expressed by `f` returning `-Inf`; the enumeration then skips `g` for that candidate. A well-posed model has at least one feasible action at every state the solver evaluates; if every action is infeasible at a node, the first action is retained as a fallback and later policy evaluation may call `g` for that pair (so `g` should tolerate it).
- `simulate` recomputes the greedy action exactly at each visited state for discrete actions (a discrete policy must not be interpolated); continuous actions keep policy interpolation.
- LQA requires a continuous action space (`ArgumentError` otherwise).

### Transition kernel (`src/transition_kernel.jl`)
- The internal `_QuadratureKernel` represents the transition kernel as finitely many weighted next-state branches per `(s, x)`. Every expectation loop routes through its primitives: `_branch_weights(ker, s, x)` (weight container, fetched once per `(s, x)`), `_branch_state(ker, s, x, j)` (next state on branch `j`; derivative-based consumers re-evaluate it at perturbed actions holding `j` fixed), and `_expected_value` (plain weighted sum of interpolant values). Do not write raw `for j in eachindex(cdp.weights)` expectation loops in new code.
- Construction: sweep-level entry points (sweeps, `s_wise_max!`, `evaluate_policy!`/`_policy_system_lu`, `evaluate!`, `simulate!`, `PolicyFunction`) call `_build_kernel(cp)` once per call; the per-state Tier-A solvers (`_s_wise_max!`, `_objective`, `_objective_and_deriv`, `_s_wise_max_foc!`, `_negH_multi!`, `_s_wise_max_multi!`, `_s_wise_max_discrete!`) receive `ker` as their second argument. For fixed weight vectors `_build_kernel` is free (no user function is evaluated); for callable weights it detects the arity (`weights(s, x)` wins if both apply) and probes one call at the first node to validate the returned container.
- `ContinuousDP.weights` may be a `Vector{Float64}` or a callable `weights(s)` / `weights(s, x)` (hybrid contract: `Tuple`/`SVector` returns keep the sweeps allocation-free, `Vector` returns are allocation-lean; weights are NOT validated to sum to 1 — sub-stochastic kernels are permitted and act as extra discounting). Action-dependent weights force the Brent fallback (the workspace drops `dfecs`); state-only weights keep the FOC path.
- Exceptions by design: LQA ignores weights (certainty-equivalent around the approximation point); `simulate!` keeps the precomputed-draws path for fixed weights and draws sequentially per step via `_draw_branch_index` for callable weights.
- The fixed-weights path must stay bit-identical to the pre-kernel code and zero-allocation; the callable static-return path must match the fixed path's sweep allocations. Both are enforced in `test/test_transition_kernel.jl`.

### Solver workspace and inner solvers (`src/cdp.jl`, `src/inner_solvers.jl`, `src/policy_system.jl`)
- `CDPWorkspace(cp::_CollocationProblem; inner_solver=:foc)` holds all preallocated buffers and evaluation caches; it is created once per `solve`. Workspaces and caches are NOT thread-safe (one per thread).
- The inner maximization over actions uses the first-order condition by default (`inner_solver=:foc`): for scalar actions, safeguarded root-finding on H'; for M-dimensional actions, box-constrained LBFGS with the analytic gradient (exact interpolant gradients, finite differences of user `f`/`g`, evaluation points clamped into the box). Automatic fallback to the derivative-free path per basis (piecewise linear bases), per state (non-finite values, exceptions), and per call (`inner_solver=:brent`): Brent for scalar actions, cyclic coordinate-wise Brent for M-dimensional ones. Warm starts live in `ws.X`.
- The scalar-action and discrete-action sweeps over states are allocation-free except for `Optim.maximize`'s small result object on Brent paths; do not introduce per-state allocations there. The `M > 1` continuous-action path is allocation-lean rather than allocation-free (small per-state bounds and optimizer work arrays; the kernel threading added one captured reference per Optim closure, ~1-2% of sweep allocations, re-baselined 2026-07); do not add avoidable allocations. Benchmark allocation columns are watched.
- `evaluate_policy!` assembles the collocation system with the point kernels, in sparse form when `Phi` is sparse (spline and piecewise linear bases). Preserve this dense/sparse dispatch.

### Out-of-domain next states (common pitfall)
Candidate actions explored during the inner maximization can map the next state outside the interpolation domain, where the fitted value function is extrapolated. For Chebyshev bases such extrapolation is astronomically large and can silently destroy a solve. Any test or benchmark model must keep `g(s, x, e)` within the domain for all feasible actions and shocks (clamp inside `g`, or tighten `x_lb`/`x_ub`). See PR #97 for background.

## Common Tasks

### Running Examples
- Examples are Jupyter notebooks in the `examples/` directory
- Cannot directly run notebooks in command line, but code can be extracted and run in Julia REPL
- Key examples: optimal growth model, Miranda & Fackler chapter 9 examples

### Algorithm Types
- **PFI (Policy Function Iteration)**: Generally faster convergence; per-iteration cost includes a collocation linear solve
- **VFI (Value Function Iteration)**: More robust but potentially slower
- **LQA (Linear Quadratic Approximation)**: Approximates the model around a reference point
- PFI and VFI are selected with `CollocationSolver(basis; algorithm=PFI)` / `CollocationSolver(basis; algorithm=VFI)`; the inner maximizer with its `inner_solver` keyword (`:foc` default, `:brent` derivative-free); LQA with `LQASolver(basis; point=point)`. The solver object is passed to `solve(cdp, solver)`.

### Basis Types
- **Chebyshev**: `ChebParams(n, s_min, s_max)` - good general purpose choice
- **Spline**: `SplineParams(breaks, s_min, s_max, k)` - flexible, good for irregular functions
- **Linear**: `LinParams(breaks, s_min, s_max)` - simple linear interpolation

### Benchmarks
- The suite in `benchmark/benchmarks.jl` follows the PkgBenchmark convention; see `benchmark/README.md` for standalone and `judge`-based usage
- Performance-related PRs must include before/after numbers from full `PkgBenchmark.benchmarkpkg` runs on the base branch and the PR branch
- A full suite run takes a few minutes. NEVER CANCEL.

### Debugging Common Issues
- **`solve` not defined**: Ensure `using ContinuousDPs` is present
- **Method convergence issues**: Try different basis sizes or algorithm (PFI vs VFI); check for out-of-domain next states (see above)
- **Simulation errors**: Check that state bounds are consistent with transition function
- **Performance issues**: Larger basis sizes increase accuracy but slow computation

## CI and Quality Assurance

### Continuous Integration
- GitHub Actions runs tests on Ubuntu, Windows, and macOS
- Tests must pass on Julia 1.x (latest stable)
- Nightly CI also runs tests on Julia nightly builds
- CompatHelper automatically updates dependency bounds

### Before Committing Changes
- ALWAYS run `julia --project=. -e "using Pkg; Pkg.test()"` to ensure all tests pass
- Verify basic functionality with a simple CDP example
- Check that examples in `examples/` directory still work if you modified core functionality
- No additional linting or formatting tools required - Julia has built-in code standards

### Pull Request Conventions
- PR and commit titles carry a category prefix: `ENH:`, `PERF:`, `BUG:`, `TST:`, `DOC:`, `MAINT:`
- Performance claims are backed by benchmark tables (see Benchmarks above)
- New numerical code paths are validated against a reference: either analytical solutions, `BasisMatrices` outputs, or a verbatim copy of the previous implementation

## Dependencies and Compatibility
- **Core dependencies**: QuantEcon.jl, BasisMatrices.jl, Optim.jl, FiniteDiff.jl, SparseArrays (stdlib)
- **Julia version**: 1.10+ required (see Project.toml)
- **Platform support**: Windows, macOS, Linux (all tested in CI)
- Dependencies install automatically via `Pkg.instantiate()`

## Performance Notes
- Basis approximation size (`n`) significantly affects both accuracy and speed
- The inner-maximization sweeps are allocation-free; VFI cost is dominated by the sweep, PFI cost by the sweep plus the collocation linear solve
- Larger shock grids increase computational complexity linearly in the sweep
- Simulation is fast once the problem is solved

## Troubleshooting
- **Long solve times**: Normal for complex problems. Increase `max_iter` if needed, but be patient
- **Convergence warnings**: Try different basis size, tolerance, or algorithm; verify next states stay within the interpolation domain
- **Memory issues**: Reduce basis size or use simpler basis types
- **Installation issues**: Ensure Julia 1.10+ and try `Pkg.update()` first
