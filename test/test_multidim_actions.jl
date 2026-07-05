using ContinuousDPs: CDPWorkspace

@testset "Multi-dimensional continuous actions" begin
    # Santos (1999, Sec. 7.3) stochastic growth model with leisure, in its
    # original two-control formulation (cf. Santos's Table 19 two-variable
    # maximization): controls x = (l, c) with l leisure and c consumption,
    # k' = z*A*k^alpha*(1-l)^(1-alpha) + (1-delta)*k - c. With delta = 1 the
    # analytical solution is the same as in test_cdp_multidim.jl: l* is
    # constant and c* = (1 - alpha*beta) * y.
    beta = 0.95
    lambda = 1 / 3
    A = 10.0
    alpha = 0.34
    delta = 1.0
    rho = 0.90

    k_min, k_max = 0.10, 10.0
    logz_min, logz_max = -0.32, 0.32

    y(k, z, l) = z * A * k^alpha * (1 - l)^(1 - alpha)

    # Controls x = (l, k'): consumption is implied by the resource
    # constraint, with infeasible pairs (c <= 0) getting reward -Inf; the
    # k' box is exactly the interpolation domain, so next states never
    # leave it (no clamp needed)
    function f2c(s, x)
        k, logz = s
        z = exp(logz)
        l, kp = x[1], x[2]
        c = y(k, z, l) + (1 - delta) * k - kp
        c > 0 || return -Inf
        return lambda * log(c) + (1 - lambda) * log(l)
    end

    g2c(s, x, e) = (x[2], rho * s[2] + e)

    x_lb2(s) = (1e-4, k_min)
    x_ub2(s) = (1 - 1e-4, k_max)

    shocks, weights = qnwnorm(7, 0.0, 0.008^2)

    # Analytical solution
    ab = alpha * beta
    l_star = (1 - lambda) * (1 - ab) /
             (lambda * (1 - alpha) + (1 - lambda) * (1 - ab))
    C_ = lambda * alpha / (1 - ab)
    D_ = lambda / ((1 - ab) * (1 - rho * beta))
    ct1 = lambda * (log(1 - ab) + log(A) + (1 - alpha) * log(1 - l_star))
    ct2 = (1 - lambda) * log(l_star)
    ct3 = beta * C_ * (log(ab) + log(A) + (1 - alpha) * log(1 - l_star))
    B_ = (ct1 + ct2 + ct3) / (1 - beta)
    v_star(k, logz) = B_ + C_ * log(k) + D_ * logz
    kp_star(k, logz) = ab * y(k, exp(logz), l_star)

    nk, nlogz = 43, 3
    dk, dz = 2, 2
    basis = Basis(SplineParams(nk - 1, k_min, k_max, dk),
                  SplineParams(nlogz - 1, logz_min, logz_max, dz))
    actions = ContinuousActions{2}(x_lb2, x_ub2)
    cdp2 = ContinuousDP(f2c, g2c, beta, shocks, weights, actions, basis)

    S = cdp2.interp.S
    v_star_on_S = v_star.(view(S, :, 1), view(S, :, 2))
    kp_star_on_S = kp_star.(view(S, :, 1), view(S, :, 2))

    # VFI x :brent is omitted: the derivative-free inner solver is
    # exercised via PFI, and VFI via :foc; combining the slowest inner
    # solver with the most outer iterations adds minutes of test time
    # without adding coverage
    @testset "solve: $method, $solver" for (method, solver) in
            ((PFI, :foc), (PFI, :brent), (VFI, :foc))
        res = solve(cdp2, method, verbose=0, max_iter=500,
                    inner_solver=solver)
        @test res.converged
        @test res.X isa Matrix{Float64}
        @test size(res.X) == (cdp2.interp.length, 2)
        @test isempty(res.X_ind)
        # Value and policy against the analytical solution. :foc
        # tolerances are in line with the one-control spline case in
        # test_cdp_multidim.jl; the derivative-free :brent path (cyclic
        # coordinate-wise Brent) is a fallback with looser policy accuracy
        # on cross-coupled objectives
        tol_l = solver == :foc ? 2e-2 : 1e-1
        tol_kp = solver == :foc ? 2e-1 : 1.0
        @test maximum(abs, res.V .- v_star_on_S) < 1.0
        @test maximum(abs, view(res.X, :, 1) .- l_star) < tol_l
        @test maximum(abs, view(res.X, :, 2) .- kp_star_on_S) < tol_kp
    end

    @testset "agreement with the one-control reduction" begin
        # The reduced model from test_cdp_multidim.jl (control = leisure,
        # consumption substituted out by the intratemporal FOC)
        c_from_l(k, z, l) = z * A * k^alpha * (1 - l)^(-alpha) *
                            (lambda / (1 - lambda)) * (1 - alpha) * l
        function f1c(s, l)
            k, logz = s
            z = exp(logz)
            (0 < l < 1) || return -Inf
            c = c_from_l(k, z, l)
            kp = y(k, z, l) + (1 - delta) * k - c
            (c <= 0 || kp < 0) && return -Inf
            return lambda * log(c) + (1 - lambda) * log(l)
        end
        function g1c(s, l, e)
            k, logz = s
            z = exp(logz)
            kp = clamp(y(k, z, l) + (1 - delta) * k - c_from_l(k, z, l),
                       k_min, k_max)
            return (kp, rho * logz + e)
        end
        cdp1 = ContinuousDP(f1c, g1c, beta, shocks, weights,
                            s -> 1e-10, s -> 1 - 1e-10, basis)
        res1 = solve(cdp1, PFI, verbose=0)
        res2 = solve(cdp2, PFI, verbose=0)
        # Same model, two formulations: the value functions must agree to
        # solver precision (approximation error is common to both)
        @test maximum(abs, res1.V .- res2.V) < 1e-4
        @test maximum(abs, res1.X .- view(res2.X, :, 1)) < 1e-3
    end

    @testset "workspace containers and warm starts" begin
        ws = CDPWorkspace(cdp2)
        @test ws.X isa Matrix{Float64}
        @test size(ws.X) == (cdp2.interp.length, 2)
        @test all(isnan, ws.X)
        @test ws.dfecs !== nothing
    end

    @testset "set_eval_nodes! and callable interface" begin
        res = solve(cdp2, PFI, verbose=0)
        k_grid = collect(range(k_min, k_max, length=15))
        logz_grid = collect(range(logz_min, logz_max, length=5))
        set_eval_nodes!(res, k_grid, logz_grid)
        @test size(res.X) == (75, 2)
        V, X, resid = res(res.eval_nodes)
        @test V ≈ res.V rtol=1e-10
        @test X ≈ res.X rtol=1e-8
    end

    @testset "simulate" begin
        res = solve(cdp2, PFI, verbose=0)
        s_path = simulate(res, [1.0, 0.0], 30)
        @test all(k_min .<= view(s_path, 1, :) .<= k_max)
        @test all(logz_min .<= view(s_path, 2, :) .<= logz_max)
    end

    @testset "construction" begin
        @test ContinuousActions{2}(x_lb2, x_ub2) isa ContinuousActions{2}
        @test_throws ArgumentError ContinuousActions{0,typeof(x_lb2),
                                                     typeof(x_ub2)}(x_lb2,
                                                                    x_ub2)
    end
end
