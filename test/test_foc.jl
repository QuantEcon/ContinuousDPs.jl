using ContinuousDPs: CDPWorkspace
using QuantEcon: qnwlogn

@testset "FOC inner solver" begin
    # 1-D stochastic optimal growth with action bounds keeping the next
    # state within the interpolation domain (as in benchmark/benchmarks.jl)
    function growth_model(basis, s_min, s_max)
        alpha = 0.65
        f(s, x) = log(x)
        g(s, x, e) = e * s^alpha - x
        shocks, weights = qnwlogn(9, 0.0, 0.01)
        e_min, e_max = extrema(shocks)
        x_lb(s) = max(1e-8, e_max * s^alpha - s_max)
        x_ub(s) = min(s, e_min * s^alpha - s_min)
        return ContinuousDP(f, g, 0.95, shocks, weights, x_lb, x_ub, basis)
    end

    s_min, s_max = 0.1, 2.0

    @testset "agreement with Brent: $label, $method" for
        (label, basis) in [
            ("Cheb", Basis(ChebParams(30, s_min, s_max))),
            ("Spline k=3", Basis(SplineParams(50, s_min, s_max, 3))),
        ],
        method in (PFI, VFI)

        cdp = growth_model(basis, s_min, s_max)
        res_foc = solve(cdp, method, verbose=0, inner_solver=:foc)
        res_brent = solve(cdp, method, verbose=0, inner_solver=:brent)
        @test res_foc.converged
        @test res_brent.converged
        @test res_foc.V ≈ res_brent.V rtol=1e-8
        @test res_foc.X ≈ res_brent.X atol=1e-6
    end

    @testset "corner solution" begin
        # H'(x) = -2(x - 2) > 0 on [0, 1]: the optimum is the upper corner
        # x = 1, with V = -(1 - 2)^2 / (1 - beta)
        beta = 0.95
        f(s, x) = -(x - 2.0)^2
        g(s, x, e) = 0.5 * s + 0.25  # stays within [0.1, 2.0]
        cdp = ContinuousDP(f, g, beta, [1.0], [1.0], s -> 0.0, s -> 1.0,
                           Basis(ChebParams(10, 0.1, 2.0)))
        res = solve(cdp, PFI, verbose=0, inner_solver=:foc)
        @test res.converged
        @test all(isapprox.(res.X, 1.0; atol=1e-6))
        # At a corner, H' != 0, so the inward evaluation offset (~sqrt(eps))
        # induces a value bias of order |H'| * offset / (1 - beta) ~ 1e-6
        @test all(isapprox.(res.V, -1.0 / (1 - beta); rtol=1e-6))
    end

    @testset "automatic Brent for non-differentiable bases" begin
        cdp = growth_model(Basis(LinParams(50, s_min, s_max)), s_min, s_max)
        ws = CDPWorkspace(cdp)  # inner_solver defaults to :foc
        @test ws.dfecs === nothing  # Lin basis: FOC unavailable
        res = solve(cdp, PFI, verbose=0)  # solves fine via Brent
        @test res.converged
    end

    @testset "argument errors" begin
        cdp = growth_model(Basis(ChebParams(10, s_min, s_max)), s_min, s_max)
        @test_throws ArgumentError solve(cdp, PFI, verbose=0,
                                         inner_solver=:newton)
        @test_throws ArgumentError CDPWorkspace(cdp, inner_solver=:bogus)
    end
end
