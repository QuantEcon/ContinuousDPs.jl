using ContinuousDPs: CDPWorkspace, bellman_operator!, policy_iteration_operator!,
                     _max_abs_diff, operator_iteration!

@testset "_max_abs_diff" begin
    @test _max_abs_diff([1.0, 2.0], [1.5, 0.5]) == 1.5
    @test _max_abs_diff([0.0], [0.0]) == 0.0
    # NaN must propagate: a diverged iteration (Inf - Inf = NaN) must not be
    # reported as converged
    @test isnan(_max_abs_diff([NaN], [NaN]))
    @test isnan(_max_abs_diff([1.0, NaN, 2.0], [1.0, NaN, 0.0]))
    @test isnan(_max_abs_diff([Inf], [Inf]))

    # End-to-end: an operator that diverges to non-finite values must not be
    # declared converged
    diverge!(C) = (C .= NaN; C)
    converged, num_iter =
        operator_iteration!(diverge!, [0.0, 0.0], 1e-8, 10, verbose=0)
    @test !converged
end

@testset "evaluate! in-place buffers" begin
    f(s, x) = log(x)
    g(s, x, e) = clamp(e * s^0.65 - x, 0.5, 2.0)
    shocks, weights = qnwnorm(5, 1.0, 0.01)
    basis = Basis(ChebParams(20, 0.5, 2.0))
    cdp = ContinuousDP(f, g, 0.95, shocks, weights, s -> 1e-8, s -> s, basis)
    res = solve(cdp, PFI, verbose=0)

    # Same-length evaluation nodes: the result arrays are reused in place
    grid1 = collect(range(0.5, 2.0, length=100))
    set_eval_nodes!(res, grid1)
    V1, X1, resid1 = res.V, res.X, res.resid
    grid2 = collect(range(0.6, 1.9, length=100))
    set_eval_nodes!(res, grid2)
    @test res.V === V1
    @test res.X === X1
    @test res.resid === resid1

    # Different length: reallocated
    set_eval_nodes!(res, collect(range(0.5, 2.0, length=50)))
    @test res.V !== V1
    @test length(res.V) == 50

    # Values agree with the callable interface at the same nodes
    V, X, resid = res(res.eval_nodes)
    @test V == res.V
    @test X == res.X
    @test resid == res.resid
end

@testset "CDPWorkspace" begin
    beta = 0.95
    f(s, x) = log(x)
    g(s, x, e) = e * s^0.65 - x
    shocks, weights = qnwnorm(5, 1.0, 0.01)
    basis = Basis(ChebParams(20, 0.5, 2.0))
    cdp = ContinuousDP(f, g, beta, shocks, weights, s -> 1e-8, s -> s, basis)

    res = solve(cdp, PFI, verbose=0)
    C0 = copy(res.C)
    n = cdp.interp.length
    ws = CDPWorkspace(cdp, inner_solver=:brent)

    @test length(ws.Tv) == n
    @test length(ws.X) == n

    # Workspace-based operators (with the Brent inner solver) agree with the
    # buffer-based ones
    Tv = Vector{Float64}(undef, n)
    @test bellman_operator!(cdp, copy(C0), ws) ==
          bellman_operator!(cdp, copy(C0), Tv)

    X = Vector{Float64}(undef, n)
    @test policy_iteration_operator!(cdp, copy(C0), ws) ==
          policy_iteration_operator!(cdp, copy(C0), X)
    @test ws.X == X
end
