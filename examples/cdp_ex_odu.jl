# Job search with learning about the offer distribution, following
# https://julia.quantecon.org/dynamic_programming/odu.html: wage offers
# are drawn from F = Beta(1, 1) or G = Beta(3, 1.2) (scaled to
# [0, w_max]); the worker does not know which. The model is solved as
# its belief MDP: the state is (w, belief) with belief = P(F), the
# Bayes update is part of the transition, and the offer distribution is
# discretized into fixed Beta-quadrature atoms whose weights are mixed
# by the current belief -- a state-dependent callable `weights`.
#
# Requires Distributions.jl in the environment besides ContinuousDPs
# and its ecosystem (QuantEcon, BasisMatrices).
using ContinuousDPs
using BasisMatrices: Basis, SplineParams
using QuantEcon: qnwbeta
using Distributions: Beta, pdf
using Random

function SearchProblem(; beta = 0.95, c = 0.6,
                       F_a = 1, F_b = 1, G_a = 3, G_b = 1.2,
                       w_max = 2.0, pi_min = 1e-3, pi_max = 1 - 1e-3,
                       quad_size = 21)
    # Beta densities scaled to [0, w_max], as in the lecture
    F, G = Beta(F_a, F_b), Beta(G_a, G_b)
    f_pdf(w) = pdf(F, w / w_max) / w_max
    g_pdf(w) = pdf(G, w / w_max) / w_max
    q(w, bel) = bel * f_pdf(w) + (1 - bel) * g_pdf(w)
    update(w, bel) = clamp(bel * f_pdf(w) / q(w, bel), pi_min, pi_max)

    # One block of Beta-quadrature atoms per candidate distribution
    # (qnwbeta embeds the density: each block's weights sum to one)
    nodes_F, qw_F = qnwbeta(quad_size, F_a, F_b)
    nodes_G, qw_G = qnwbeta(quad_size, G_a, G_b)
    shocks = w_max .* vcat(nodes_F, nodes_G)
    weights(s) = vcat(s[2] .* qw_F, (1 - s[2]) .* qw_G)

    f(s, x) = x === :accept ? s[1] : c
    g(s, x, wp) = x === :accept ? (s[1], s[2]) : (wp, update(wp, s[2]))

    return (; beta, c, w_max, pi_min, pi_max, F, G, update,
            f, g, shocks, weights)
end

sp = SearchProblem()

cdp = ContinuousDP(f=sp.f, g=sp.g, discount=sp.beta,
                   actions=DiscreteActions([:reject, :accept]),
                   shocks=sp.shocks, weights=sp.weights)

basis = Basis(
    SplineParams(collect(range(0.0, sp.w_max, length=40)), 0, 3),
    SplineParams(collect(range(sp.pi_min, sp.pi_max, length=40)), 0, 3))
res = solve(cdp, CollocationSolver(basis); verbose=0)
println("converged: ", res.converged)

# Reservation wage: smallest acceptable offer at a given belief, from
# the greedy policy evaluated at user-supplied states via res(s_nodes)
w_grid = collect(range(1e-3, sp.w_max, length=400))
function w_bar(belief)
    _, X, _ = res(hcat(w_grid, fill(belief, length(w_grid))))
    return w_grid[findfirst(==(:accept), X)]
end
println("w_bar: ", [round(w_bar(b), digits=3) for b in 0.1:0.2:0.9])

# Simulation: a path of (w_t, belief_t) under the greedy policy, with
# offers drawn from the worker's belief-mixed subjective distribution
# (the callable-weights path of `simulate`). Acceptance is absorbing, so
# the path freezes at the accepted (wage, belief).
seed = 42
rng = MersenneTwister(seed)
path = simulate(rng, res, [1.0, 0.5], 30)   # 2 x 30: w row, belief row
t_acc = findfirst(t -> path[1, t] == path[1, t + 1], 1:29)
println("accepted at t = ", t_acc,
        ",  wage = ", round(path[1, t_acc + 1], digits=3),
        ",  belief = ", round(path[2, t_acc + 1], digits=3))
