# Job search with learning about the offer distribution, following
# https://julia.quantecon.org/dynamic_programming/odu.html: wage offers
# are drawn from F = Beta(1, 1) or G = Beta(3, 1.2) (scaled to
# [0, w_max]); the worker does not know which. The model is the belief
# MDP with state (w, belief), belief = P(F), solved by the Bellman
# equation collocation method via POMDPs.solve(CollocationSolver(basis), m).
#
# The model is defined twice — as an explicit problem type, then as a
# QuickMDP — and the two solutions are cross-checked at the end.
#
# Requires POMDPs, POMDPTools, QuickPOMDPs, and Distributions besides
# ContinuousDPs and its ecosystem (QuantEcon, BasisMatrices). Note that
# ContinuousDPs and POMDPs both export `solve` and `simulate`: with
# both loaded, qualify the calls. (For the same model through the
# native ContinuousDP interface, see examples/cdp_ex_odu.jl.)
using ContinuousDPs
using POMDPs, POMDPTools, QuickPOMDPs
using BasisMatrices: Basis, SplineParams
using QuantEcon: qnwbeta
using Distributions: Beta, pdf
using Random

#= The model as an explicit problem type =#

struct SearchMDP <: POMDPs.MDP{NTuple{2,Float64},Symbol}
    beta::Float64
    c::Float64
    w_max::Float64
    pi_min::Float64
    pi_max::Float64
    F::Beta{Float64}
    G::Beta{Float64}
    shocks::Vector{Float64}  # offer nodes: Beta quadrature, F then G block
    qw_F::Vector{Float64}    # block weights (each block sums to one)
    qw_G::Vector{Float64}
end

function SearchMDP(; beta=0.95, c=0.6, F_a=1, F_b=1, G_a=3, G_b=1.2,
                   w_max=2.0, pi_min=1e-3, pi_max=1 - 1e-3,
                   quad_size=21)
    F, G = Beta(F_a, F_b), Beta(G_a, G_b)
    nodes_F, qw_F = qnwbeta(quad_size, F_a, F_b)
    nodes_G, qw_G = qnwbeta(quad_size, G_a, G_b)
    return SearchMDP(beta, c, w_max, pi_min, pi_max, F, G,
                     w_max .* vcat(nodes_F, nodes_G), qw_F, qw_G)
end

f_pdf(m::SearchMDP, w) = pdf(m.F, w / m.w_max) / m.w_max
g_pdf(m::SearchMDP, w) = pdf(m.G, w / m.w_max) / m.w_max
q(m::SearchMDP, w, bel) = bel * f_pdf(m, w) + (1 - bel) * g_pdf(m, w)
bayes(m::SearchMDP, w, bel) =
    clamp(bel * f_pdf(m, w) / q(m, w, bel), m.pi_min, m.pi_max)

POMDPs.actions(::SearchMDP) = (:reject, :accept)
POMDPs.discount(m::SearchMDP) = m.beta
POMDPs.reward(m::SearchMDP, s, x) = x === :accept ? s[1] : m.c

function POMDPs.transition(m::SearchMDP, s, x)
    w, bel = s
    x === :accept && return Deterministic((w, bel))   # absorbing
    # Belief-mixed offer distribution, Bayes update in the next state
    return SparseCat([(wp, bayes(m, wp, bel)) for wp in m.shocks],
                     vcat(bel .* m.qw_F, (1 - bel) .* m.qw_G))
end

POMDPs.initialstate(::SearchMDP) = Deterministic((1.0, 0.5))

#= Solve; the basis domain is the approximation domain =#

m = SearchMDP()
basis = Basis(
    SplineParams(collect(range(0.0, m.w_max, length=40)), 0, 3),
    SplineParams(collect(range(m.pi_min, m.pi_max, length=40)), 0, 3))
policy = POMDPs.solve(CollocationSolver(basis), m)
println("converged: ", policy.res.converged)

# Reservation wage: smallest acceptable offer at a given belief
w_grid = collect(range(1e-3, m.w_max, length=400))
w_bar(policy, bel) =
    w_grid[findfirst(w -> action(policy, (w, bel)) === :accept, w_grid)]
println("w_bar: ",
        [round(w_bar(policy, b), digits=3) for b in 0.1:0.2:0.9])

# Rollout under the solved policy; report the first accepted offer
hist = POMDPs.simulate(
    HistoryRecorder(max_steps=30, rng=MersenneTwister(42)), m, policy)
for (t, step) in enumerate(eachstep(hist))
    if step.a === :accept
        println("accepted at t = ", t,
                ",  wage = ", round(step.s[1], digits=3),
                ",  belief = ", round(step.s[2], digits=3))
        break
    end
end

#= The same model as a QuickMDP =#

# Built inside functions so the closures capture locals, not globals
function SearchProblem(; beta = 0.95, c = 0.6,
                       F_a = 1, F_b = 1, G_a = 3, G_b = 1.2,
                       w_max = 2.0, pi_min = 1e-3, pi_max = 1 - 1e-3,
                       quad_size = 21)
    F, G = Beta(F_a, F_b), Beta(G_a, G_b)
    f_pdf(w) = pdf(F, w / w_max) / w_max
    g_pdf(w) = pdf(G, w / w_max) / w_max
    q(w, bel) = bel * f_pdf(w) + (1 - bel) * g_pdf(w)
    update(w, bel) = clamp(bel * f_pdf(w) / q(w, bel), pi_min, pi_max)
    nodes_F, qw_F = qnwbeta(quad_size, F_a, F_b)
    nodes_G, qw_G = qnwbeta(quad_size, G_a, G_b)
    shocks = w_max .* vcat(nodes_F, nodes_G)
    return (; beta, c, w_max, pi_min, pi_max, update, shocks, qw_F, qw_G)
end

function search_mdp(sp)
    (; beta, c, update, shocks, qw_F, qw_G) = sp
    return QuickMDP(
        statetype  = NTuple{2,Float64},          # (offer w, belief)
        actiontype = Symbol,
        actions    = [:reject, :accept],
        discount   = beta,
        reward     = (s, x) -> x === :accept ? s[1] : c,
        transition = function (s, x)
            w, bel = s
            x === :accept && return Deterministic((w, bel))
            SparseCat([(wp, update(wp, bel)) for wp in shocks],
                      vcat(bel .* qw_F, (1 - bel) .* qw_G))
        end,
        initialstate = Deterministic((1.0, 0.5)),
    )
end

policy_quick = POMDPs.solve(CollocationSolver(basis),
                            search_mdp(SearchProblem()))

# Same model, two definition styles: the solutions must agree
wbars = [w_bar(policy, b) for b in 0.1:0.2:0.9]
wbars_quick = [w_bar(policy_quick, b) for b in 0.1:0.2:0.9]
println("reservation wages agree: ", wbars == wbars_quick)
println("max coefficient difference: ",
        maximum(abs, policy.res.C - policy_quick.res.C))
