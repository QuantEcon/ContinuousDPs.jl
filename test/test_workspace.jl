using ContinuousDPs: CDPWorkspace, bellman_operator!, policy_iteration_operator!

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
    ws = CDPWorkspace(cdp)

    @test length(ws.Tv) == n
    @test length(ws.X) == n

    # Workspace-based operators agree with the buffer-based ones
    Tv = Vector{Float64}(undef, n)
    @test bellman_operator!(cdp, copy(C0), ws) ==
          bellman_operator!(cdp, copy(C0), Tv)

    X = Vector{Float64}(undef, n)
    @test policy_iteration_operator!(cdp, copy(C0), ws) ==
          policy_iteration_operator!(cdp, copy(C0), X)
    @test ws.X == X
end
