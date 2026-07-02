#=
Benchmark suite for ContinuousDPs.jl

Defines `SUITE` in the standard BenchmarkTools format, usable with
PkgBenchmark.jl or AirspeedVelocity.jl.

To run standalone:

    julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
    julia --project=benchmark benchmark/benchmarks.jl

Covers the main computational kernels and end-to-end solves:
- `_s_wise_max!` (per-state optimization kernel)
- `bellman_operator!` / `compute_greedy!` (state loops over the kernel)
- `evaluate_policy!` (PFI linear solve)
- `set_eval_nodes!` (evaluation on non-interpolation grids)
- end-to-end `solve` with PFI and (iteration-capped) VFI
=#
using BenchmarkTools
using ContinuousDPs
using ContinuousDPs:
    _s_wise_max!, bellman_operator!, compute_greedy!, evaluate_policy!
using BasisMatrices: Basis, ChebParams, SplineParams
using QuantEcon: qnwlogn, qnwnorm

#= Model definitions =#

# 1-D stochastic optimal growth
function growth_model_1d(basis)
    alpha = 0.65
    f(s, x) = log(x)
    g(s, x, e) = e * s^alpha - x
    shocks, weights = qnwlogn(9, 0.0, 0.01)
    x_lb(s) = 1e-8
    x_ub(s) = s
    return ContinuousDP(f, g, 0.95, shocks, weights, x_lb, x_ub, basis)
end

# 2-D stochastic optimal growth with leisure (Santos, 1999, Sec. 7.3;
# same model as in test/test_cdp_multidim.jl)
function growth_model_2d(basis)
    beta = 0.95
    lambda = 1 / 3
    A = 10.0
    alpha = 0.34
    delta = 1.0
    rho = 0.90

    y(k, z, x) = z * A * k^alpha * (1 - x)^(1 - alpha)
    function c_from_x(k, z, x)
        term1 = z * A * k^alpha * (1 - x)^(-alpha)
        term2 = (lambda / (1 - lambda)) * (1 - alpha) * x
        return term1 * term2
    end
    kprime_from_x(k, z, x) = y(k, z, x) + (1 - delta) * k - c_from_x(k, z, x)

    function f(s, x)
        k, logz = s
        z = exp(logz)
        (0 < x < 1) || return -Inf
        c = c_from_x(k, z, x)
        kp = kprime_from_x(k, z, x)
        (c <= 0 || kp < 0) && return -Inf
        return lambda * log(c) + (1 - lambda) * log(x)
    end

    function g(s, x, e)
        k, logz = s
        z = exp(logz)
        kp = kprime_from_x(k, z, x)
        logzp = rho * logz + e
        return (kp, logzp)
    end

    shocks, weights = qnwnorm(7, 0.0, 0.008^2)
    x_lb(s) = 1e-10
    x_ub(s) = 1 - 1e-10
    return ContinuousDP(f, g, beta, shocks, weights, x_lb, x_ub, basis)
end

#= Benchmark cases =#

s_min_1d, s_max_1d = 0.1, 2.0
k_min, k_max = 0.10, 10.0
logz_min, logz_max = -0.32, 0.32
nk, nlogz = 43, 3

cases = [
    ("1d_cheb", growth_model_1d(Basis(ChebParams(50, s_min_1d, s_max_1d)))),
    ("1d_spline", growth_model_1d(
        Basis(SplineParams(99, s_min_1d, s_max_1d, 3)))),
    ("2d_spline", growth_model_2d(
        Basis(SplineParams(nk - 1, k_min, k_max, 2),
              SplineParams(nlogz - 1, logz_min, logz_max, 2)))),
]

eval_grids = Dict(
    "1d_cheb" => (collect(range(s_min_1d, s_max_1d, length=500)),),
    "1d_spline" => (collect(range(s_min_1d, s_max_1d, length=500)),),
    "2d_spline" => (collect(range(k_min, k_max, length=30)),
                    collect(range(logz_min, logz_max, length=10))),
)

#= Suite =#

const SUITE = BenchmarkGroup()

for (label, cdp) in cases
    grp = SUITE[label] = BenchmarkGroup()

    # End-to-end solves
    grp["solve_PFI"] = @benchmarkable solve($cdp, PFI; verbose=0)
    grp["solve_VFI_50iter"] =
        @benchmarkable solve($cdp, VFI; max_iter=50, verbose=0)

    # Kernel benchmarks at the converged solution, for realistic inputs
    res = solve(cdp, PFI; verbose=0)
    C0 = copy(res.C)
    X0 = copy(res.X)
    n = cdp.interp.length
    N = ndims(cdp)

    # Per-state optimization kernel (#73)
    s_mid = N == 1 ? cdp.interp.S[div(n, 2)] : cdp.interp.S[div(n, 2), :]
    sp = Matrix{Float64}(undef, size(cdp.shocks, 1), N)
    grp["s_wise_max_one_state"] =
        @benchmarkable _s_wise_max!($cdp, $s_mid, $C0, $sp)

    # State loops over the kernel
    grp["bellman_operator"] = @benchmarkable bellman_operator!(
        $cdp, C, Tv
    ) setup = (C = copy($C0); Tv = Vector{Float64}(undef, $n)) evals = 1
    grp["compute_greedy"] = @benchmarkable compute_greedy!(
        $cdp, $C0, X
    ) setup = (X = Vector{Float64}(undef, $n)) evals = 1

    # Policy evaluation: matrix assembly + LU solve (#75)
    grp["evaluate_policy"] = @benchmarkable evaluate_policy!(
        $cdp, $X0, C
    ) setup = (C = Vector{Float64}(undef, $n)) evals = 1

    # Evaluation on a non-interpolation grid (#74)
    grid = eval_grids[label]
    grp["set_eval_nodes"] =
        @benchmarkable set_eval_nodes!($res, $grid...) evals = 1
end

#= Standalone execution =#

if abspath(PROGRAM_FILE) == @__FILE__
    tune!(SUITE)
    results = run(SUITE; verbose=true)
    show(IOContext(stdout, :compact => false), MIME"text/plain"(), results)
    println()
end
