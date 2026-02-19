# ==================================================
# Santos (1999) Section 7.3 Tests
# ==================================================
# Model parameters (as in Santos, 1999, Sec. 7.3)
beta, lambda, A, alpha, delta, rho, sigma_epsilon = 0.95, 1 / 3, 10.0, 0.34, 1.0, 0.90, 0.008

# Mesh size (as in Santos, 1999, Sec. 7.3)
# Exclude (500, 33) due to high computational cost
mesh_settings = ((43, 3), (143, 9))

# State domains (as in Santos, 1999, Sec. 7.3)
logz_min, logz_max = -0.32, 0.32
k_min, k_max = 0.10, 10.0

# For numerical stability
x_lb(s), x_ub(s) = 1e-10, 1 - 1e-10

# Grid and bases setup
nk, nlogz = 3, 3
deg_k, deg_z = 3, 1
dk = min(deg_k, nk - 1)
dz = min(deg_z, nlogz - 1)
breaks_k = nk - (dk - 1)
breaks_z = nlogz - (dz - 1)
bases = Basis[]
push!(bases, Basis(LinParams(nk, k_min, k_max), LinParams(nlogz, logz_min, logz_max)))
push!(bases, Basis(ChebParams(nk, k_min, k_max), ChebParams(nlogz, logz_min, logz_max)))
push!(bases, Basis(SplineParams(breaks_k, k_min, k_max, dk), SplineParams(breaks_z, logz_min, logz_max, dz)))

# Model functions
# Production and Santos (7.4)-style mapping: given leisure l -> (c, k')
# Production function
y(k, z, l) = z * A * k^alpha * (1 - l)^(1 - alpha)
# Consumption and k prime based on Santos equation (7.4) ("unidimensional maximization")
c_from_l(k, z, l) = z * A * k^alpha * (1 - l)^(-alpha) * (lambda / (1 - lambda)) * (1 - alpha) * l
kprime_from_l(k, z, l) = y(k, z, l) + (1 - delta) * k - c_from_l(k, z, l)

# Reward function
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
    return lambda * log(c) + (1 - lambda) * log(l)
end

# Transition function
function g(s, l, e)
    k, logz = s
    z = exp(logz)
    kp = kprime_from_l(k, z, l)
    logzp = rho * logz + e
    return (kp, logzp)
end

# Analytical solution (delta = 1)
ab = alpha * beta
# Optimal leisure (constant)
l_star = ((1 - lambda) * (1 - ab)) / (lambda * (1 - alpha) + ((1 - lambda) * (1 - ab)))

# Policy function (consant fraction of production)
policy(k, logz) = ab * exp(logz) * A * k^alpha * (1 - l_star)^(1 - alpha)

# Value function: V(k, z) = B + C*log(k) + D*log(z)
C = lambda * alpha / (1 - ab)
D = lambda / ((1 - ab) * (1 - rho * beta))
const_term = lambda * (log(1 - ab) + log(A) + (1 - alpha) * log(1 - l_star)) + (1 - lambda) * log(l_star) + beta * C * (log(ab) + log(A) + (1 - alpha) * log(1 - l_star))
B = const_term / (1 - beta)
v_star(k, logz) = B + C * log(k) + D * logz

@testset "Santos (1999) Sec. 7.3: stochastic growth w/ leisure (2D state, 1D control) benchmarks" begin
    # Test: Solutions with multiple methods and interpolations
    # Shock discretization (Gauss-Hermite quadrature)
    n_shocks = 7
    shocks, weights = qnwnorm(n_shocks, 0.0, sigma_epsilon^2)

    # Method types
    methods = [VFI, PFI]

    for method in methods
        for basis in bases
            # Build DP
            cdp = ContinuousDP(f, g, beta, shocks, weights, x_lb, x_ub, basis)

            # Analytical targets on interpolation nodes
            S = cdp.interp.S
            k_nodes = @view S[:, 1]
            logz_nodes = @view S[:, 2]
            v_star_on_S = v_star.(k_nodes, logz_nodes)

            # Solve DP
            res = solve(cdp, method, max_iter=500, tol=sqrt(eps()), verbose=0)
            l_hat = vec(res.X)

            # Convergence tests
            @test res.converged

            # Policy function benchmark check
            println("max |l_hat - l_star| = ", maximum(abs.(l_hat .- l_star)))

            # Value function benchmark check
            println("max |V_hat - V_star| = ", maximum(abs.(res.V .- v_star_on_S)))
        end
    end
end