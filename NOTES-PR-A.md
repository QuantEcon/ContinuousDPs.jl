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
   optgrowth notebook present the keyword constructor + solver types. Checked
   2026-07-17: `cdp_ex_MF_jl.ipynb` uses the deprecated 8-arg constructor 4 times
   (`lqapprox_jl.ipynb` is clean). Decided (tentatively): update the MF notebook
   in PR A.
4. ~~Benchmarks~~ DONE: solves via `CollocationSolver` (end-to-end timings now include
   per-solve `Interp` construction — expect a small constant addition on `solve_*`
   keys in the judge report), kernels via `_with_interp`.
5. Version stays 0.2.1 in this PR (per review): the 0.3.0 bump happens in a separate
   release PR. Release notes go there.
6. ~~`PkgBenchmark.judge` vs `main`~~ DONE (target a0b186c vs baseline bbd2b26,
   Julia 1.12.6, M4 Max; full table in the session scratchpad `judge_report.md` —
   paste into the PR body). Summary: **all internal kernels are unchanged** (time
   ratios 0.97-1.06 within the 10% tolerance, memory 1.00). The `solve_*` keys show
   the expected per-solve `Interp` construction now included in the timed call:
   time is neutral everywhere except `1d_spline solve_PFI` (1.26x — a
   milliseconds-scale solve where building + factorizing the 101-node sparse basis
   matrix is a visible fraction), and memory ratios on `solve_*` are up by one
   Interp construction (large *ratios* like 8.5x on VFI-50 correspond to small
   absolute baselines, since main prepaid Interp at problem construction). If this
   ever matters for re-solve-heavy workflows, the option is a lazily cached Interp
   inside CollocationSolver (revisits D6; costs statelessness/thread-safety) —
   recommend accepting as is and documenting.
   NOTE: PkgBenchmark cannot run in the working repo (LibGit2 chokes on
   `.claude/worktrees/`); run it from a clean clone.
7. When the PR is complete: delete this file AND remove it from the git history.
   DECIDED (2026-07-17): **history rewrite + force-push**, deferred to the last
   minute before merge — e.g. `git rebase -i main` dropping the file from each
   commit, or
   `git filter-repo --invert-paths --path NOTES-PR-A.md --refs collocation-solver`.
   The branch is already pushed, so this needs a force-push to the PR branch
   (normal for PR branches). (Squash-merge would also keep the file out of
   `main`'s history, but the rewrite route was preferred.)
   NOT reasonable after merge: that would mean rewriting `main` and force-pushing a
   shared branch. If the file ever lands on `main`, leave it and just delete it.

## Decisions from the 2026-07-17 review session (PR B-facing; none change PR A code)

- **Point-evaluation API (for PR B's `CollocationPolicy`): functor types**
  `ValueFunction(res)` and `PolicyFunction(res)` — small structs holding
  preallocated caches, callable as `(vf::ValueFunction)(s)` / `(pf::PolicyFunction)(s)`.
  Unexported (semi-public, documented). Semantics: `ValueFunction` evaluates the
  fitted V via `funeval_point!(fec, res.C, s)`; `PolicyFunction` is exact greedy
  for discrete actions and interpolate-and-clamp for continuous actions —
  exactly the policy closure `simulate!` builds internally today, factored out
  (`simulate!` then constructs a `PolicyFunction`; existing simulation tests
  validate the refactor). Refinement: evaluate the policy interpolant via a
  second `FunEvalCache` (the point kernels support `LinParams`) instead of
  `Interpoland`, making both functors allocation-free per call; `simulate!`
  inherits the speedup. Thread-safety caveat as `CDPWorkspace` (one per thread).
  **Lands in PR B** (non-breaking 0.3.x addition), not PR A. Rationale: cheap
  per-step `action`/`value` for POMDPs rollouts without reaching through
  `res.cdp.interp`; alternatives (per-call use of the `res(s_nodes)` callable —
  2-3 orders of magnitude slower; keyword modes on the callable; exposing raw
  kernels) rejected. Future option on `PolicyFunction`: exact-greedy mode for
  continuous actions via a constructor keyword.
- **`CDPMDP` typing for discrete actions** (Float64-only vs parametric action
  type): postponed to PR B.
- **`POMDPs.initialstate`**: leave undefined when not supplied (simulators take
  explicit starts); open to a later change.
- **PR B docs caveat wording**: ContinuousDPs extends `QuantEcon.solve` (it does
  not own `solve`), so the name-clash caveat is `QuantEcon.solve` vs
  `POMDPs.solve`; the clash predates this package pair. Also: with dual-trigger
  weakdeps `[POMDPs, POMDPTools]`, the `as_mdp` MethodError hint must name both
  packages.
- **PR C mechanism** (explicit-finite contract vs `weighted_iterator`-consuming
  kernel refactor): left open.
- **Housekeeping** (post design comment to #89, cross-ref on #8, note on
  QuantEcon/QuantEcon.jl#398): deferred.

## Environment note

The (untracked) `Manifest.toml` was stale for Julia 1.12 (FFTW load failure);
`Pkg.update()` fixed it. Not a code issue.
