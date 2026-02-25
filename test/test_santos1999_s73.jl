# ==================================================
# Santos (1999) Section 7.3 Tests
# ==================================================
using Random

# Model parameters (as in Santos, 1999, Sec. 7.3)
beta, lambda, A, alpha, delta, rho, sigma_epsilon = 0.95, 1 / 3, 10.0, 0.34, 1.0, 0.90, 0.008

# Random seed for reproducibility
seed = 1234

# State domains (as in Santos, 1999, Sec. 7.3)
logz_min, logz_max = -0.32, 0.32
k_min, k_max = 0.10, 10.0

# For numerical stability
x_lb(s) = 1e-10
x_ub(s) = 1 - 1e-10

# Mesh size (as in Santos, 1999, Sec. 7.3)
# Exclude (143, 9) and (500, 33) due to high computational cost
mesh_setting = (43, 3)
nk, nlogz = mesh_setting
grid_size_k = (k_max - k_min) / (nk - 1)
grid_size_logz = (logz_max - logz_min) / (nlogz - 1)
mesh_size_h = sqrt(grid_size_k^2 + grid_size_logz^2)

# Model functions
# Production and Santos (7.4)-style mapping: given leisure x -> (c, k')
# Production function
y(k, z, x) = z * A * k^alpha * (1 - x)^(1 - alpha)
# Consumption and k prime based on Santos equation (7.4) ("unidimensional maximization")
c_from_x(k, z, x) = z * A * k^alpha * (1 - x)^(-alpha) * (lambda / (1 - lambda)) * (1 - alpha) * x
kprime_from_x(k, z, x) = y(k, z, x) + (1 - delta) * k - c_from_x(k, z, x)

# Reward function
function f(s, x)
    k, logz = s
    z = exp(logz)
    if !(0 < x < 1)
        return -Inf
    end
    c = c_from_x(k, z, x)
    kp = kprime_from_x(k, z, x)
    if c <= 0 || kp < 0
        return -Inf
    end
    return lambda * log(c) + (1 - lambda) * log(x)
end

# Transition function
function g(s, x, e)
    k, logz = s
    z = exp(logz)
    kp = kprime_from_x(k, z, x)
    logzp = rho * logz + e
    return (kp, logzp)
end

# Analytical solution (delta = 1)
ab = alpha * beta
# Optimal leisure (constant)
x_star = ((1 - lambda) * (1 - ab)) / (lambda * (1 - alpha) + ((1 - lambda) * (1 - ab)))

# Policy function (constant fraction of production)
policy(k, logz) = ab * exp(logz) * A * k^alpha * (1 - x_star)^(1 - alpha)

# Value function: V(k, z) = B + C*log(k) + D*log(z)
C = lambda * alpha / (1 - ab)
D = lambda / ((1 - ab) * (1 - rho * beta))
const_term = lambda * (log(1 - ab) + log(A) + (1 - alpha) * log(1 - x_star)) + (1 - lambda) * log(x_star) + beta * C * (log(ab) + log(A) + (1 - alpha) * log(1 - x_star))
B = const_term / (1 - beta)
v_star(k, logz) = B + C * log(k) + D * logz

@testset "Multidimensional-state stochastic optimal growth" begin
    # Test: Linear basis with VFI and PFI should match analytical solution on interpolation nodes within tolerances based on Santos (1999) Sec. 7.3
    # Shock discretization (Gauss-Hermite quadrature)
    n_shocks = 7
    shocks, weights = qnwnorm(n_shocks, 0.0, sigma_epsilon^2)

    # Method types
    methods = [VFI, PFI]

    for method in methods
        # Calculate mesh size for tolerance settings
        # Tolerances based on Santos (1999) Table 16: observed constants are approximately 0.36
        policy_tol = 0.4 * mesh_size_h
        # Tol based on Santos (1999) Table 16: measured upper bounds are approximately 24
        value_tol = 24 * mesh_size_h^2

        # Build basis
        basis = Basis(LinParams(nk, k_min, k_max), LinParams(nlogz, logz_min, logz_max))

        # Build DP
        cdp = ContinuousDP(f, g, beta, shocks, weights, x_lb, x_ub, basis)

        # Analytical targets on interpolation nodes
        S = cdp.interp.S
        k_nodes = @view S[:, 1]
        logz_nodes = @view S[:, 2]
        v_star_on_S = v_star.(k_nodes, logz_nodes)
        k_prime_star_on_S = policy.(k_nodes, logz_nodes)

        # Solve DP
        res = solve(cdp, method, max_iter=500, tol=sqrt(eps()), verbose=0)
        x_hat = vec(res.X)
        k_hat = kprime_from_x.(k_nodes, exp.(logz_nodes), x_hat)

        # Convergence tests
        @test res.converged

        # Policy function benchmark check
        @test maximum(abs, k_hat .- k_prime_star_on_S) <= policy_tol

        # Value function benchmark check
        @test maximum(abs, res.V .- v_star_on_S) <= value_tol

        # set_eval_nodes! 
        k_grid = collect(range(k_min, k_max, length=15))
        logz_grid = collect(range(logz_min, logz_max, length=7))
        @test_nowarn set_eval_nodes!(res, k_grid, logz_grid)

        # simulate
        s_init = [0.1, 0.0]
        ts_length = 50
        rng = MersenneTwister(seed)
        s_path = simulate(rng, res, s_init, ts_length)
        k_path = @view s_path[1, :]
        logz_path = @view s_path[2, :]

        # Check if k stays within bounds
        @test all(k_path .>= k_min) && all(k_path .<= k_max)

        # Check if logz stays within bounds
        @test all(logz_path .>= logz_min) && all(logz_path .<= logz_max)

    end
end
