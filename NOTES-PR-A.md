# PR A working notes — primitives-only `ContinuousDP` + `CollocationSolver` (closes #8)

**Status: core implemented and green.** All 100% of the pre-existing test suite passes
through the deprecation shims (equivalence check), plus a smoke test covering the new
API, old-vs-new exact agreement (PFI and LQA), error paths, and `@inferred` on the new
`solve`. This file is deleted before merge.

## Context

Design of record: the #89 issue thread (POMDPs.jl interface) — the plan there sequences
PR A (this branch, closes #8) before PR B (`ext/ContinuousDPsPOMDPsExt.jl` with
`as_mdp`/`CDPMDP`/`CollocationPolicy`) and optional PR C (generic explicit-finite
POMDPs models). PR A is v0.3.0 (breaking, with a one-release deprecation cycle).

## New public API

```julia
growth = ContinuousDP(                      # keyword form is canonical
    f = (s, x) -> log(x),
    g = (s, x, e) -> (s - x)^0.4 * e,
    discount = 0.96,                        # `discount`, not `beta`
    shocks = shocks, weights = weights,
    x_lb = s -> 1e-5, x_ub = s -> s,        # or: actions = ContinuousActions/DiscreteActions
)
solver = CollocationSolver(basis; algorithm = PFI, inner_solver = :foc,
                           tol = sqrt(eps()), max_iter = 500)
res = solve(growth, solver; v_init = ..., verbose = 2, print_skip = 50)
res_lqa = solve(growth, LQASolver(basis; point = (s, x, e)))
```

Positional primitives forms also exist (`ContinuousDP(f, g, discount, shocks, weights,
actions_or_xlb, [x_ub])`). Exports gained `CollocationSolver`, `LQASolver`.

## Implementation approach (important for reviewers/continuation)

**Rebind, not thread-through.** `ContinuousDP` keeps its `interp` field but the bound is
now `TI <: Union{Interp{N}, Nothing}`; primitives-only problems carry `interp = nothing`
with sentinel `N = 0` (`ndims` throws an informative error for `{0}`). The new
`solve(cdp, solver)` builds `Interp(solver.basis)` per call and *rebinds* via
`_with_interp(cdp, interp)` (explicit type parameters — the implicit constructor cannot
infer `N` when `TI = Nothing`), then runs the pre-existing pipeline (`_solve_core`,
formerly the body of `solve`). **Every internal function (sweeps, operators,
`evaluate_policy!`, `evaluate!`, `set_eval_nodes!`, `simulate!`, LQA `_solve!`,
`CDPWorkspace`) is untouched** — they keep reading `cdp.interp` from the rebound
instance. This is why old results match exactly and why `benchmark/benchmarks.jl`
signatures remain valid.

Deprecated (one release, via `Base.depwarn`): the 8-arg/7-arg basis-endowed
constructors, the `basis` kwarg of the copy constructor, and
`solve(cdp, PFI/VFI/LQA; ...)` (which now guards `interp === nothing` with an
instructive `ArgumentError` and forwards to `_solve_core`). LQA note: it *uses* the
basis (fits its value function via `Phi_lu`), hence `LQASolver` carries one; `point`
moved from a `solve` kwarg into `LQASolver`.

## Remaining work (in order)

1. **New test file** (`test/test_solver_types.jl` or fold into `test_cdp.jl`): adapt
   `scratchpad/smoke_pr_a.jl` (see git stash/scratch or rewrite: keyword-constructor
   variants and exclusivity errors, `CollocationSolver`/`LQASolver` construction and
   validation errors, old-vs-new exact agreement on PFI/VFI/LQA, `v_init` length check,
   `ndims` error, `@inferred`, copy-constructor interp semantics). Register in
   `test/runtests.jl`.
2. **Migrate existing tests to the new API** (old API coverage shrinks to one dedicated
   deprecation testset). Files: `test_cdp.jl`, `test_cdp_multidim.jl`, `test_foc.jl`,
   `test_evaluate_policy.jl`, `test_workspace.jl`, `test_lq_approx.jl` construct via the
   old constructor; internal-function tests can use `ContinuousDPs._with_interp`.
3. **Docs sync** (per repo instructions, README + `docs/src/index.md` +
   `examples/cdp_ex_optgrowth_jl.ipynb` are kept content-synchronized): rewrite the
   problem-formulation/interface sections around primitives + solver. Also
   `docs/make.jl` API page if it lists docstrings.
4. **Benchmarks**: `benchmark/benchmarks.jl` — switch `solve` benches to the new API;
   internal-operator benches need `_with_interp` for a bound instance.
5. **Version bump** to 0.3.0 in `Project.toml`; release notes summarizing the
   deprecations.
6. **Full `PkgBenchmark.judge`** vs `main` (expect neutral; the only new per-solve cost
   is `Interp` construction, which the old constructor paid at problem construction).
7. Delete this file before merge.

## Environment note

The (untracked) `Manifest.toml` was stale for Julia 1.12 (FFTW load failure);
`Pkg.update()` fixed it. Not a code issue.
