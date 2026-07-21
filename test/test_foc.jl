using ContinuousDPs: CDPWorkspace, set_coefs!, _s_wise_max_foc!,
                     _s_wise_max!, _build_kernel
using QuantEcon: qnwlogn

@testset "FOC inner solver" begin
    # 1-D stochastic optimal growth with action bounds keeping the next
    # state within the interpolation domain (as in benchmark/benchmarks.jl)
    function growth_model(s_min, s_max)
        alpha = 0.65
        f(s, x) = log(x)
        g(s, x, e) = e * s^alpha - x
        shocks, weights = qnwlogn(9, 0.0, 0.01)
        e_min, e_max = extrema(shocks)
        x_lb(s) = max(1e-8, e_max * s^alpha - s_max)
        x_ub(s) = min(s, e_min * s^alpha - s_min)
        return ContinuousDP(f, g, 0.95, shocks, weights, x_lb, x_ub)
    end

    s_min, s_max = 0.1, 2.0
    cdp = growth_model(s_min, s_max)

    @testset "agreement with Brent: $label, $method" for
        (label, basis) in [
            ("Cheb", Basis(ChebParams(30, s_min, s_max))),
            ("Spline k=3", Basis(SplineParams(50, s_min, s_max, 3))),
        ],
        method in (PFI, VFI)

        res_foc = solve(cdp, CollocationSolver(basis; algorithm=method,
                                               inner_solver=:foc), verbose=0)
        res_brent = solve(cdp, CollocationSolver(basis; algorithm=method,
                                                 inner_solver=:brent),
                          verbose=0)
        @test res_foc.converged
        @test res_brent.converged
        @test res_foc.V ≈ res_brent.V rtol=1e-8
        @test res_foc.X ≈ res_brent.X atol=1e-6
    end

    @testset "direct interior FOC/Brent agreement: $label" for (label, basis) in [
        ("Cheb", Basis(ChebParams(30, s_min, s_max))),
        ("Spline k=3", Basis(SplineParams(50, s_min, s_max, 3))),
    ]
        # Compare the two inner solvers state by state at the converged
        # coefficients: the solve-level comparison cannot see the FOC
        # maximizers directly, since `evaluate!` recomputes `res.X` with
        # Brent
        res = solve(cdp, CollocationSolver(basis; inner_solver=:foc),
                    verbose=0)
        colloc_cdp = ContinuousDPs._colloc(res)
        ws = CDPWorkspace(colloc_cdp)
        foreach(dfec -> set_coefs!(dfec, res.C), ws.dfecs)
        ker = _build_kernel(colloc_cdp)
        for i in 1:colloc_cdp.interp.length
            s = colloc_cdp.interp.S[i]
            v_foc, x_foc = _s_wise_max_foc!(colloc_cdp.cdp, ker, s, res.C,
                                            ws.fec, ws.dfecs, NaN)
            v_brent, x_brent = _s_wise_max!(colloc_cdp.cdp, ker, s, res.C,
                                            ws.fec)
            @test v_foc ≈ v_brent rtol=1e-8
            @test x_foc ≈ x_brent atol=1e-6
        end
    end

    @testset "corner solution" begin
        # H'(x) = -2(x - 2) > 0 on [0, 1]: the optimum is the upper corner
        # x = 1, with V = -(1 - 2)^2 / (1 - beta)
        beta = 0.95
        f(s, x) = -(x - 2.0)^2
        g(s, x, e) = 0.5 * s + 0.25  # stays within [0.1, 2.0]
        cdp_c = ContinuousDP(f, g, beta, [1.0], [1.0], s -> 0.0, s -> 1.0)
        res = solve(cdp_c, CollocationSolver(Basis(ChebParams(10, 0.1, 2.0));
                                             inner_solver=:foc), verbose=0)
        @test res.converged
        @test all(isapprox.(res.X, 1.0; atol=1e-6))
        # At a corner, H' != 0, so the inward evaluation offset (~sqrt(eps))
        # induces a value bias of order |H'| * offset / (1 - beta) ~ 1e-6
        @test all(isapprox.(res.V, -1.0 / (1 - beta); rtol=1e-6))
    end

    @testset "corner solution with f finite but wrong outside the bounds" begin
        # f is finite everywhere but drops sharply just outside the feasible
        # set [0, 1]: the finite-difference step for f_x must not cross the
        # bound, or the spurious out-of-bounds values would push the
        # computed maximizer ~1e-5 inside the corner. Probe the FOC path
        # directly (the Brent-based `evaluate!` would mask it in `res.X`).
        beta = 0.95
        f(s, x) = 0.0 <= x <= 1.0 ? -(x - 2.0)^2 : -100.0
        g(s, x, e) = 0.5 * s + 0.25
        cdp_c = ContinuousDP(f, g, beta, [1.0], [1.0], s -> 0.0, s -> 1.0)
        res = solve(cdp_c, CollocationSolver(Basis(ChebParams(10, 0.1, 2.0));
                                             inner_solver=:foc), verbose=0)
        @test res.converged
        colloc_cdp = ContinuousDPs._colloc(res)
        ws = CDPWorkspace(colloc_cdp)
        foreach(dfec -> set_coefs!(dfec, res.C), ws.dfecs)
        v, x = _s_wise_max_foc!(colloc_cdp.cdp, _build_kernel(colloc_cdp),
                                colloc_cdp.interp.S[1],
                                res.C, ws.fec, ws.dfecs, NaN)
        @test x ≈ 1.0 atol=1e-6
    end

    @testset "automatic Brent for non-differentiable bases" begin
        basis = Basis(LinParams(50, s_min, s_max))
        res = solve(cdp, CollocationSolver(basis), verbose=0)  # solves fine via Brent
        @test res.converged
        ws = CDPWorkspace(ContinuousDPs._colloc(res))  # inner_solver defaults to :foc
        @test ws.dfecs === nothing  # Lin basis: FOC unavailable
    end

    @testset "argument errors" begin
        basis = Basis(ChebParams(10, s_min, s_max))
        @test_throws ArgumentError CollocationSolver(basis;
                                                     inner_solver=:newton)
        colloc_cdp = ContinuousDPs._CollocationProblem(cdp, ContinuousDPs.Interp(basis))
        @test_throws ArgumentError CDPWorkspace(colloc_cdp, inner_solver=:bogus)
    end
end
