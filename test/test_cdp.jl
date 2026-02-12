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

        for basis in bases
            cdp = ContinuousDP(f, g, beta, shocks, weights, x_lb, x_ub, basis)

            for method in methods
                # solve
                tol = sqrt(eps())
                max_iter = 500
                res = @inferred(solve(cdp, method, tol=tol, max_iter=max_iter))

                rtol = 1e-5
                @test isapprox(res.V, v_star.(cdp.interp.S); rtol=rtol)
                @test isapprox(res.X, x_star.(cdp.interp.S); rtol=rtol)

                # set_eval_nodes!
                grid_size = 200
                eval_nodes = collect(range(s_min, stop=s_max, length=grid_size))
                set_eval_nodes!(res, eval_nodes);

                # simulate
                s_path = @inferred(simulate(res, s_init, ts_length))
                atol = 1e-5
                @test isapprox(s_path[end], s_path_star[end]; atol=atol)
            end
        end

        @testset "Test warning" begin
            cdp = ContinuousDP(
                f, g, beta, shocks, weights, x_lb, x_ub, bases[1]
            )
            for max_iter in [0, 1]
                @test_logs (:warn, r".*max_iter.*")
                           solve(cdp, max_iter=max_iter)
           end
        end
    end

    @testset "Santos (1999) Sec. 7.3: stochastic growth w/ leisure (2D state, 1D control) benchmarks" begin
        # Model parameters (as in Santos, 1999, Sec. 7.3)
        struct Santos1999Params
            beta::Float64
            gamma::Float64
            A::Float64
            alpha::Float64
            delta::Float64
            rho::Float64
            sigma_epsilon::Float64

            function Santos1999params(beta, gamma, A, alpha, delta, rho, sigma_epsilon)
                @assert 0 < beta < 1 "beta must be in (0,1)"
                @assert 0 < gamma <= 1 "gamma must be in (0,1]"
                @assert A > 0 "A must be positive "
                @assert 0 < alpha < 1 "alpha must be in (0,1)"
                @assert 0 <= delta <= 1 "delta must be in [0,1]"
                @assert 0 <= rho < 1 "rho must be in [0, 1)"
                @assert sigma_epsilon >= 0 "sigma_epsilon must be non-negative"
                new(beta, gamma, A, alpha, delta, rho, sigma_epsilon)
            end
        end
        
        """
        Default parameters from Santos (1999) Section 7.3, Table 16
        """
        default_params() = Santos1999Params(0.95, 1/3, 10.0, 0.34, 1.0, 0.90, 0.008)

        # State domains (as in Santos, 1999, Sec. 7.3)
        logz_min, logz_max = -0.32, 0.32
        k_min, k_max = 0.10, 10.0

        # For numerical stability
        x_lb(s), x_ub(s) = 1e-10, 1 - 1e-10
        
        # Model functions
        # Production and Santos (7.4)-style mapping: given leisure l -> (c, k')
        # Output
        function build_production(params::Santos1999Params)
            function y(k, z, l)
                return z * params.A * k^params.alpha * (1 - l)^(1 - params.alpha)
            end
        end

        # Calculate consumption and k prime based on Santos equation (7.4) ("unidimensional maximization")
        function build_c_from_l(params::Santos1999Params)
            function c_from_l(k, z, l)
                return z * params.A * k^params.alpha * (1 - l)^(-params.alpha) * (params.lambda / (1 - params.lambda)) * (1 - params.alpha) * l
            end
        end

        function build_kprime_from_l(params::Santos1999Params)
            y = build_production(params)
            c_from_l = build_c_from_l(params)

            function kprime_from_l(k, z, l)
                return y(k, z, l) + (1 - params.delta) * k - c_from_l(k, z, l)
            end
        end

        # Reward function
        function build_reward_function(params::Santos73Params)
            c_from_l = build_c_from_l(params)
            kprime_from_l = build_kprime_from_l(params)

            function f(s, l) 
                k, logz = s
                z = exp(logz)
                if !(0 < l < 1)
                    return -Inf
                end
                c = c_from_l(k, z, l)
                kp = kprime_from_l(k, z, l)
                if c <= 0 || kp < 0
                    return -Inf
                end
                return params.lambda*log(c) + (1 - params.lambda)*log(l)
            end
        end

        # Transition function
        function build_transition_function(params::Santos73Params)
            kprime_from_l = build_kprime_from_l(params)

            function g(s, l, e)
                k, logz = s
                z = exp(logz)
                kp = kprime_from_l(k, z, l)
                logzp = params.rho*logz + e
                return (kp, logzp)
            end
        end

        # Analytical solution (delta = 1)
        function analytical_solution(params::Santos73Params)
            @assert params.delta == 1.0 "Analytical solution is only for delta = 1"

            ab = params.alpha * params.beta

            # Optimal leisure (constant)
            l_star = ((1 - params.lambda)*(1 - ab)) / (params.lambda*(1 - params.alpha) + ((1 - params.lambda)*(1 - ab)))
            
            # Value function: V(k, z) = B + C*log(k) + D*log(z)
            C = params.lambda * params.alpha / (1 - ab)
            D = params.lambda / ((1 - ab) * (1 - params.rho * beta))
            
            const_term = params.lambda * (log(1 - ab) + log(A) + (1 - params.alpha) * log(1 - l_star)) + (1 - params.lambda) * log(l_star) + params.beta * C * (log(ab) + log(A) + (1-params.alpha) * log(1 - l_star))
            B = const_term / (1 - beta)

            # Policy function (consant fraction of production)
            policy(k, logz) = ab & exp(logz) * params.A * k^params.alpha * (1 - l_star)^(1 - params.alpha)

            return B, C, D, l_star, policy
        end

        # Tests
        # Test 1: Parameter construction
	    @testset "Parameter construction and validation" begin
            # Test valid parameters
            @test_nowarn Santos1999Params(0.95, 1/3, 10.0, 0.34, 1.0, 0.90, 0.008)

            # Test default parameters
            params = default_params()
            @test params.beta == 0.95
            @test params.gamma == 1/3
            @test params.delta == 1.0

            # Test invalid parameters
            # beta >= 1
            @test_throws AssertionError Santos1999Params(1.5, 1/3, 10.0, 0.34, 1.0, 0.90, 0.008)
            # gamma < 0
            @test_throws AssertionError Santos1999Params(0.95, -0.3, 10.0, 0.34, 1.0, 0.90, 0.008)
	    end

        # Test 2: Analytical solution (delta = 1) properties
        @testset "Analytical solution propoerties" begin
            params = default_params()
            B, C, D, l_star, policy = analytical_solution(params)

            # Check coefficients
            @test C > 0
            @test D > 0
            @test 0 < l_star < 1

        end

        # Test 3: Solutions with multiple methods and interpolations
        @testset "Solutions: methods x interpolations" begin
            params = default_params()

            # Grid setup
            nk, nlogz = 5, 5

            # Shock discretization (Gauss-Hermite quadrature)
            n_shocks = 7
            shocks, weights = qnwnorm(n_shocks, 0.0, params.sigma_epsilon^2)

            # Build functions
            f = build_reward_function(params)
            g = build_transition_function(params)

            # Constuct analytical solutions
            B, C, D, l_star, policy = analytical_solution(params)
            v_star(k, logz) = B + C * log(k) + D * logz

            # Method types
            methods = [VFI, PFI]
            method_names = ["VFI", "PFI"]

            # Interpolation types
            interp_types = [
                ("Linear", () -> Basis(LinParams(nk, k_min, k_max), LinParams(nlogz, logz_min, logz_max)))
                ("Spline", () -> Basis(SplineParams(collect(range(k_min, k_max, length=nk)), 0, 3), SplineParams(collect(range(logz_min, logz_max, length=nlogz)), 0, 3)))
                ("Chebyshev", () -> Basis(ChebParams(nk, k_min, k_max), ChebParams(nlogz, logz_min, logz_max)))
            ]

            # Test each combination
            results = Dict()

            for (method, method_names) in zip(method, method_names)
                for (interp_name, basis_builder) in interp_types
                    test_name = "$method_name + $interp_name"

                    @testset "$test_name" begin
                        # Build interpolation basis
                        basis = basis_builder()
                        
                        # Build DP
                        cdp = ContinuousDP(f, g, params.beta, shocks, weights, x_lb, x_ub, basis)
                        
                        # Analytical targets on interpolation nodes
                        S = cdp.interp.S
                        v_star_on_S = [v_star(row) for row in eachrow(S)]
                        k_prime_star_on_S = [policy(row) for row in eachrow(S)]
                        
                        # Solve DP
                        res = solve(cdp, method, max_iter=500, tol=sqrt(eps()))
                        results[test_name] = res
                        l_hat = vec(res.X)

                        # Convergence tests
                        @test res.converged
                        @test res.iter < 500

                        # Value function finite test
                        @test all(isfinite.(res.Vf))

                        # Policy (leisure) shold be in bounds
                        @test all (x_lb .<= res.X .<= x_ub)

                        # 
                        println("=== $test_name vs Santos(1999) analytical solution benchmark (delta = 1) ===")
                        # Iteration number check
                        println("$test_name: converged in $(res.iter) iterations")

                        # Policy function benchmark check
                        println("Analytical l* = ", l_star)
                        println("l_hat range on interpolation nodes: [", minimum(l_hat), ", ", maximum(l_hat), "]" )
                        println("max |l_hat - l_star| = ", maximum(abs.(l_hat .- l_star)))

                        # Value function benchmark check
                        println("max |V_hat - V_star| = ", maximum(abs.(res.Vf .- v_star_on_S)))
                    end
                end
            end

            # Compare methods
            @testset "Cross-method consistency" begin
                if haskey(results, "VFI + Chebyshev") && haskey(rsults, "PFI + Chebyshev")
                    res_vfi = results["VFI + Chebyshev"]
                    res_pfi = results["PFI + Chebyshev"]

                    # Policy function difference test
                    pf_diff = maximum(abs.(res_vfi.X - res_pfi.X))
                    @test pf_diff < 0.1

                    # Value function difference test
                    vf_diff = maximum(abs.(res_vfi.Vf - res_pfi.Vf))
                    @test vf_diff < 0.5

                    println("VFI vs PFI : max policy diff = $pf_diff, max value diff = $vf_diff")
                end
            end
        end
    end


    @testset "LQ control" begin
        using QuantEcon

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

        cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, basis)

        res_lqa = @inferred(solve(cdp, LQA, point=point));
        rtol = 1e-2

        @test isapprox(res_lqa.V, v_star.(cdp.interp.S); rtol=rtol)
        @test isapprox(res_lqa.X, x_star.(cdp.interp.S); rtol=rtol)

    end

    @testset "Initial value" begin
        # Construct optimal growth model
        n = 10
        s_min, s_max = 5, 10
        basis = Basis(LinParams(n, s_min, s_max))

        alpha = 0.2
        bet = 0.5
        gamm = 0.9
        sigma = 0.1
        discount = 0.9;

        x_star = ((discount * bet) / (1 - discount * gamm))^(1 / (1 - bet))
        s_star = gamm * x_star + x_star^bet
        s_star, x_star

        f(s, x) = (s - x)^(1 - alpha) / (1 - alpha)
        g(s, x, e) = gamm * x .+ e * x^bet;

        n_shocks = 3
        shocks, weights = zeros(3), ones(3) / 3.

        x_lb(s) = 0
        x_ub(s) = 0.99 * s;

        cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, basis)

        # Compute coefficients once
        v_init = π * ones(n)
        res = solve(cdp, v_init=v_init, max_iter=0)

        # Basis is identity matrix
        @test isapprox(res.C, v_init)

    end

end
