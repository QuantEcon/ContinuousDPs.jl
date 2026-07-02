# ContinuousDPs.jl Benchmarks

This directory contains a benchmark suite in the standard
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) format:
[`benchmarks.jl`](benchmarks.jl) defines a `BenchmarkGroup` named `SUITE`,
which can be run standalone or through
[PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl).

## What is benchmarked

The suite runs three model cases:

- `1d_cheb`: 1-D stochastic optimal growth, Chebyshev basis (50 nodes,
  9 shock nodes);
- `1d_spline`: same model, cubic spline basis (101 nodes);
- `2d_spline`: 2-D stochastic optimal growth with leisure
  (Santos, 1999, Sec. 7.3; same model as in `test/test_cdp_multidim.jl`),
  quadratic spline basis (43 × 3 nodes, 7 shock nodes).

For each case, the following are benchmarked:

| Key | Description |
|:----|:------------|
| `s_wise_max_one_state` | Per-state maximization kernel `_s_wise_max!` |
| `bellman_operator` | One application of `bellman_operator!` |
| `compute_greedy` | One application of `compute_greedy!` |
| `evaluate_policy` | One application of `evaluate_policy!` |
| `set_eval_nodes` | Evaluation on a non-interpolation grid |
| `solve_PFI` | End-to-end `solve` with PFI |
| `solve_VFI_50iter` | `solve` with VFI, capped at 50 iterations |

Kernel benchmarks use basis coefficients from the converged PFI solution so
that inputs are realistic. VFI is capped at `max_iter=50` so that the
benchmark measures a fixed amount of work independently of convergence
behavior.

## Running the suite standalone

From the repository root, set up the benchmark environment once:

```
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run the whole suite (takes a few minutes):

```
julia --project=benchmark benchmark/benchmarks.jl
```

To run interactively, e.g., only a subset:

```julia
julia> include("benchmark/benchmarks.jl");

julia> run(SUITE["1d_cheb"]["bellman_operator"])
```

## Running with PkgBenchmark.jl

Install PkgBenchmark in your default environment, then, with this package
active (e.g. `julia --project=.`):

```julia
using PkgBenchmark

results = benchmarkpkg("ContinuousDPs")
export_markdown("results.md", results)
```

### Comparing two commits

To evaluate the performance change of a target commit (or branch) relative
to a baseline:

```julia
jud = judge("ContinuousDPs", "<target>", "<baseline>")
export_markdown("judgement.md", jud)
```

For example, to compare the current state of `main` against the previous
commit:

```julia
jud = judge("ContinuousDPs", "main", "main~1")
```

Display the judgment summary:

```julia
julia> show(PkgBenchmark.benchmarkgroup(jud))
3-element BenchmarkTools.BenchmarkGroup:
  tags: []
  "1d_cheb" => 7-element BenchmarkTools.BenchmarkGroup:
      tags: []
      "bellman_operator" => TrialJudgement(-35.20% => improvement)
      "s_wise_max_one_state" => TrialJudgement(-33.87% => improvement)
      ...
```

and the timing estimates of each side:

```julia
julia> show(jud.baseline_results.benchmarkgroup)

julia> show(jud.target_results.benchmarkgroup)
```

Note that `judge` checks out and runs each commit, so uncommitted changes in
the working tree are not included.

For comprehensive usage details, refer to the
[PkgBenchmark documentation](https://juliaci.github.io/PkgBenchmark.jl/stable).
