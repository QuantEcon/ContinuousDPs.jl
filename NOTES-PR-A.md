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

## Planned future removal of the `interp` field (v0.4+, agreed 2026-07-17)

The `interp::Union{Interp{N},Nothing}` field is deprecation-era structure. When the
deprecated basis-endowed constructors are removed (v0.4), do the threading refactor as
a separate `MAINT:` PR behind the migrated test suite: `ContinuousDP{Tf,Tg,TR,TA}`
loses the field **and the `N` parameter** (with it the `N = 0` sentinel); internals
move to an internal bound type (e.g. `_BoundCDP{N}` wrapping `(cdp, interp)`) or take
`interp` explicitly; `CDPSolveResult` holds `cdp` and `interp` as separate fields.
Signature changes will touch the semi-public operators tracked by
`benchmark/benchmarks.jl`. Forward-compat rule for PR B: the POMDPs extension must not
reach through `res.cdp.interp` — consume only the public surface of `CDPSolveResult`.

## Remaining work

1. ~~New test file~~ DONE: `test/test_solver_types.jl` (58 tests incl. the deprecation
   testset with exact old-vs-new agreement).
2. ~~Migrate existing tests~~ DONE: all files construct primitives-only problems;
   internal-function tests bind via `_with_interp`. Full suite green (842 tests).
3. ~~Docs sync~~ DONE: README, `docs/src/index.md`, `docs/src/api.md`, and the
   optgrowth notebook present the keyword constructor + solver types. The MF and
   lqapprox notebooks were NOT touched (not part of the synchronized triple) — check
   them before release and update if they construct `ContinuousDP` with a basis.
4. ~~Benchmarks~~ DONE: solves via `CollocationSolver` (end-to-end timings now include
   per-solve `Interp` construction — expect a small constant addition on `solve_*`
   keys in the judge report), kernels via `_with_interp`.
5. Version stays 0.2.1 in this PR (per review): the 0.3.0 bump happens in a separate
   release PR. Release notes go there.
6. **`PkgBenchmark.judge` vs `main`**: launched; report at
   `scratchpad/judge_report.md` (re-run if lost: `judge(path, "collocation-solver",
   "main")` with PkgBenchmark in a temp env). Paste the table into the PR body.
7. When the PR is complete: delete this file AND remove it from the git history.
   Removing it from history is possible and reasonable as long as it happens *before*
   the branch lands on `main`:
   - Easiest: **squash-merge the PR** after deleting the file in a final commit — the
     squashed commit's tree is all that reaches `main`, so the file and the
     intermediate commits never enter `main`'s history. No history surgery needed.
   - If a merge-commit/rebase-merge workflow is preferred: rewrite the branch first,
     e.g. `git rebase -i main` dropping the file from each commit, or
     `git filter-repo --invert-paths --path NOTES-PR-A.md --refs collocation-solver`.
     Safe now (the branch has not been pushed); after pushing, it just needs a
     force-push to the PR branch (normal for PR branches).
   - NOT reasonable after merge: that would mean rewriting `main` and force-pushing a
     shared branch. If the file ever lands on `main`, leave it and just delete it.

## Environment note

The (untracked) `Manifest.toml` was stale for Julia 1.12 (FFTW load failure);
`Pkg.update()` fixed it. Not a code issue.
