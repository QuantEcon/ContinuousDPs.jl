@testset "cdp.jl" begin

    @testset "Deterministic optimal growth" begin
        alpha = 0.65
        beta = 0.95
        f(s, x) = log(x)
        g(s, x, e) = s^alpha - x
        shocks = [1.]
        weights = [1.]
        x_lb(s) = 0
        x_ub(s) = s

        cdp = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks,
                           weights=weights, x_lb=x_lb, x_ub=x_ub)

        # Analytical solution
        ab = alpha * beta
        c1 = (log(1 - ab) + log(ab) * ab / (1 - ab)) / (1 - beta)
        c2 = alpha / (1 - ab)
        v_star(s) = c1 + c2 * log(s)
        x_star(s) = (1 - ab) * s^alpha

        # Analytical state path
        s_init = 0.1
        ts_length = 25
        s_path_star = Array{Float64}(undef, ts_length)
        s_path_star[1] = s_init
        shock = 1.  # Arbitrary
        for t in 1:ts_length-1
            s_path_star[t+1] = g(s_path_star[t], x_star(s_path_star[t]), shock)
        end

        # Bases
        bases = Basis[]

        # Chebyshev
        n = 30
        s_min, s_max = 0.1, 2.
        push!(bases, Basis(ChebParams(n, s_min, s_max)))

        # Spline
        k = 3
        m = 101
        breaks = m - (k-1)
        s_min, s_max = 0.1, 2.
        push!(bases, Basis(SplineParams(breaks, s_min, s_max, k)))

        methods = [PFI, VFI]
        basis_labels = ["Chebyshev", "Spline"]

        for (basis, label) in zip(bases, basis_labels)
            for method in methods
                @testset "Test $method with $label basis" begin
                    # solve
                    tol = sqrt(eps())
                    max_iter = 500
                    solver = CollocationSolver(basis; algorithm=method,
                                               tol=tol, max_iter=max_iter)
                    res = @inferred(solve(cdp, solver))

                    rtol = 1e-5
                    @test isapprox(res.V, v_star.(res.eval_nodes); rtol=rtol)
                    @test isapprox(res.X, x_star.(res.eval_nodes); rtol=rtol)

                    # set_eval_nodes!
                    grid_size = 200
                    eval_nodes = collect(range(s_min, stop=s_max, length=grid_size))
                    set_eval_nodes!(res, eval_nodes);

                    # simulate
                    s_path = @inferred(simulate(res, s_init, ts_length))
                    atol = 1e-4
                    @test s_path[1] == s_init
                    @test length(s_path) == ts_length
                    @test maximum(abs, s_path - s_path_star) <= atol
                end
            end
        end

        @testset "Test warning" begin
            for max_iter in [0, 1]
                solver = CollocationSolver(bases[1]; max_iter=max_iter)
                @test_logs (:warn, r".*max_iter.*") solve(cdp, solver;
                                                          verbose=1)
           end
        end

        @testset "Test type inference" begin
            interp_fact(res) = res.cdp.interp.Phi_lu
            transition_fun(res) = res.cdp.g
            interp_basis(res) = res.cdp.interp.basis

            res = @inferred solve(cdp, CollocationSolver(bases[1]))

            @inferred interp_fact(res)
            @inferred transition_fun(res)
            @inferred interp_basis(res)
        end
    end

    @testset "LQ control" begin
        import QuantEcon

        A = [1.0 0.0;
             -0.5 0.9];

        R = [5.0 0.0;
             0.0 0.3]

        N = [0.05 0.1]

        Q = 0.1;

        C = 0.0;

        B = [0.0; 1.5];

        f(s, x) = -([1, s...]' * R * [1, s...] .+ x' * Q * x .+
                    2 * x' * N * [1, s...])[1];
        g(s, x, e) = (A * [1, s...] + B * x)[2];

        point = (5.0, 0.0, 0.0);

        discount = 0.9;
        lq = QuantEcon.LQ(Q, R, A, B, C, N, bet=discount);
        P, F, d = stationary_values(lq);
        v_star(s) = -([1, s...]' * P * [1, s...] + d)
        x_star(s) = -(F * [1, s...])[1];

        n = 100
        s_min, s_max = -5.0, 10.
        basis = Basis(LinParams(n, s_min, s_max))

        x_lb(s) = -20.0
        x_ub(s) = 5.0;

        shocks = [0.]
        weights = [1.]

        cdp = ContinuousDP(f=f, g=g, discount=discount, shocks=shocks,
                           weights=weights, x_lb=x_lb, x_ub=x_ub)

        res_lqa = @inferred(solve(cdp, LQASolver(basis; point=point)));
        rtol = 1e-2

        @test isapprox(res_lqa.V, v_star.(res_lqa.eval_nodes); rtol=rtol)
        @test isapprox(res_lqa.X, x_star.(res_lqa.eval_nodes); rtol=rtol)

    end

    @testset "Initial value" begin
        # Construct optimal growth model
        n = 10
        s_min, s_max = 5, 10
        basis = Basis(LinParams(n, s_min, s_max))

        alpha = 0.2
        bet = 0.5
        gamm = 0.9
        discount = 0.9;

        f(s, x) = (s - x)^(1 - alpha) / (1 - alpha)
        g(s, x, e) = gamm * x .+ e * x^bet;

        n_shocks = 3
        shocks, weights = zeros(n_shocks), ones(n_shocks) / n_shocks

        x_lb(s) = 0
        x_ub(s) = 0.99 * s;

        cdp = ContinuousDP(f=f, g=g, discount=discount, shocks=shocks,
                           weights=weights, x_lb=x_lb, x_ub=x_ub)

        # Compute coefficients once
        v_init = π * ones(n)
        res = solve(cdp, CollocationSolver(basis; max_iter=0); v_init=v_init,
                    verbose=0)

        # Basis is identity matrix
        @test isapprox(res.C, v_init)

    end

end
