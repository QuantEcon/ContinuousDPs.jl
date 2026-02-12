# ==================================================
# Santos (1999) Section 7.3 Tests
# ==================================================
# Model parameters (as in Santos, 1999, Sec. 7.3)
struct Santos1999Params
    beta::Float64
    lambda::Float64
    A::Float64
    alpha::Float64
    delta::Float64
    rho::Float64
    sigma_epsilon::Float64

    function Santos1999Params(beta, lambda, A, alpha, delta, rho, sigma_epsilon)
        @assert 0 < beta < 1 "beta must be in (0,1)"
        @assert 0 < lambda <= 1 "lambda must be in (0,1]"
        @assert A > 0 "A must be positive "
        @assert 0 < alpha < 1 "alpha must be in (0,1)"
        @assert 0 <= delta <= 1 "delta must be in [0,1]"
        @assert 0 <= rho < 1 "rho must be in [0, 1)"
        @assert sigma_epsilon >= 0 "sigma_epsilon must be non-negative"
        new(beta, lambda, A, alpha, delta, rho, sigma_epsilon)
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

    return y
end

# Calculate consumption and k prime based on Santos equation (7.4) ("unidimensional maximization")
function build_c_from_l(params::Santos1999Params)
    function c_from_l(k, z, l)
        return z * params.A * k^params.alpha * (1 - l)^(-params.alpha) * (params.lambda / (1 - params.lambda)) * (1 - params.alpha) * l
    end

    return c_from_l
end

function build_kprime_from_l(params::Santos1999Params)
    y = build_production(params)
    c_from_l = build_c_from_l(params)

    function kprime_from_l(k, z, l)
        return y(k, z, l) + (1 - params.delta) * k - c_from_l(k, z, l)
    end

    return kprime_from_l
end

# Reward function
function build_reward_function(params::Santos1999Params)
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

    return f
end

# Transition function
function build_transition_function(params::Santos1999Params)
    kprime_from_l = build_kprime_from_l(params)

    function g(s, l, e)
        k, logz = s
        z = exp(logz)
        kp = kprime_from_l(k, z, l)
        logzp = params.rho*logz + e
        return (kp, logzp)
    end

    return g
end

# Analytical solution (delta = 1)
function analytical_solution(params::Santos1999Params)
    @assert params.delta == 1.0 "Analytical solution is only for delta = 1"

    ab = params.alpha * params.beta

    # Optimal leisure (constant)
    l_star = ((1 - params.lambda)*(1 - ab)) / (params.lambda*(1 - params.alpha) + ((1 - params.lambda)*(1 - ab)))
    
    # Value function: V(k, z) = B + C*log(k) + D*log(z)
    C = params.lambda * params.alpha / (1 - ab)
    D = params.lambda / ((1 - ab) * (1 - params.rho * params.beta))
    
    const_term = params.lambda * (log(1 - ab) + log(params.A) + (1 - params.alpha) * log(1 - l_star)) + (1 - params.lambda) * log(l_star) + params.beta * C * (log(ab) + log(params.A) + (1-params.alpha) * log(1 - l_star))
    B = const_term / (1 - params.beta)

    # Policy function (consant fraction of production)
    policy(k, logz) = ab * exp(logz) * params.A * k^params.alpha * (1 - l_star)^(1 - params.alpha)

    # Value function
    v_star(k, logz) = B + C * log(k) + D * logz

    return B, C, D, l_star, policy, v_star
end


@testset "Santos (1999) Sec. 7.3: stochastic growth w/ leisure (2D state, 1D control) benchmarks" begin
    # Tests
    # Test 1: Parameter construction
    @testset "Parameter construction and validation" begin
        # Test valid parameters
        @test_nowarn Santos1999Params(0.95, 1/3, 10.0, 0.34, 1.0, 0.90, 0.008)

        # Test default parameters
        params = default_params()
        @test params.beta == 0.95
        @test params.lambda == 1/3
        @test params.delta == 1.0

        # Test invalid parameters
        # beta >= 1
        @test_throws AssertionError Santos1999Params(1.5, 1/3, 10.0, 0.34, 1.0, 0.90, 0.008)
        # lambda < 0
        @test_throws AssertionError Santos1999Params(0.95, -0.3, 10.0, 0.34, 1.0, 0.90, 0.008)
    end

    # Test 2: Analytical solution (delta = 1) properties
    @testset "Analytical solution propoerties" begin
        params = default_params()
        B, C, D, l_star, policy, v_star = analytical_solution(params)

        # Check coefficients
        @test C > 0
        @test D > 0
        @test 0 < l_star < 1

    end

    # Test 3: Solutions with multiple methods and interpolations
    @testset "Solutions: methods x interpolations" begin
        params = default_params()

        # Grid setup
        nk, nlogz = 3, 3

        # Shock discretization (Gauss-Hermite quadrature)
        n_shocks = 7
        shocks, weights = qnwnorm(n_shocks, 0.0, params.sigma_epsilon^2)

        # Build functions
        f = build_reward_function(params)
        g = build_transition_function(params)

        # Constuct analytical solutions
        B, C, D, l_star, policy, v_star = analytical_solution(params)

        # Method types
        methods = [VFI, PFI]
        method_names = ["VFI", "PFI"]

        # Interpolation types
        interp_types = [
            ("Linear", () -> Basis(LinParams(nk, k_min, k_max), LinParams(nlogz, logz_min, logz_max))),
            ("Spline", () -> Basis(SplineParams(collect(range(k_min, k_max, length=nk)), 0, 3), SplineParams(collect(range(logz_min, logz_max, length=nlogz)), 0, 3))),
            ("Chebyshev", () -> Basis(ChebParams(nk, k_min, k_max), ChebParams(nlogz, logz_min, logz_max)))
        ]

        # Test each combination
        results = Dict()

        for (method, method_name) in zip(methods, method_names)
            for (interp_name, basis_builder) in interp_types
                test_name = "$method_name + $interp_name"

                @testset "$test_name" begin
                    # Build interpolation basis
                    basis = basis_builder()
                    
                    # Build DP
                    cdp = ContinuousDP(f, g, params.beta, shocks, weights, x_lb, x_ub, basis)
                    
                    # Analytical targets on interpolation nodes
                    S = cdp.interp.S
                    k_nodes = @view S[:, 1]
                    logz_nodes = @view S[:, 2]
                    v_star_on_S = v_star.(k_nodes, logz_nodes)
                    k_prime_star_on_S = policy.(k_nodes, logz_nodes)
                    
                    # Solve DP
                    res = solve(cdp, method, max_iter=500, tol=sqrt(eps()), verbose=0)
                    results[test_name] = res
                    l_hat = vec(res.X)

                    # Convergence tests
                    @test res.converged
                    @test res.num_iter < 500

                    # Value function finite test
                    @test all(isfinite.(res.V))

                    # Policy (leisure) shold be in bounds
                    lb = x_lb(S[1, :])
                    ub = x_ub(S[1, :])
                    @test all((lb .<= res.X) .& (res.X .<= ub))

                    println("=== $test_name vs Santos(1999) analytical solution benchmark (delta = 1) ===")
                    # Iteration number check
                    println("$test_name: converged in $(res.num_iter) iterations")

                    # Policy function benchmark check
                    println("Analytical l* = ", l_star)
                    println("l_hat range on interpolation nodes: [", minimum(l_hat), ", ", maximum(l_hat), "]" )
                    println("max |l_hat - l_star| = ", maximum(abs.(l_hat .- l_star)))

                    # Value function benchmark check
                    println("max |V_hat - V_star| = ", maximum(abs.(res.V .- v_star_on_S)))
                end
            end
        end

        # Compare methods
        @testset "Cross-method consistency" begin
            if haskey(results, "VFI + Chebyshev") && haskey(results, "PFI + Chebyshev")
                res_vfi = results["VFI + Chebyshev"]
                res_pfi = results["PFI + Chebyshev"]

                # Policy function difference test
                pf_diff = maximum(abs.(res_vfi.X - res_pfi.X))
                @test pf_diff < 0.1

                # Value function difference test
                vf_diff = maximum(abs.(res_vfi.V - res_pfi.V))
                @test vf_diff < 0.5

                println("VFI vs PFI : max policy diff = $pf_diff, max value diff = $vf_diff")
            end
        end
    end
end