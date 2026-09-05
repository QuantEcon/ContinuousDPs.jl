# The household problem of the Aiyagari model, following
# https://julia.quantecon.org/multi_agent_models/aiyagari.html, defined
# through the POMDPs.jl interface -- the same model specification as in
# QuantEcon/QuantEcon.jl#405, where it is solved on an asset grid by
# DiscreteDP -- and solved here by the Bellman equation collocation
# method with CONTINUOUS assets and a CONTINUOUS savings choice: the
# feasible action set at each state is an `ActionInterval`.
#
# The model is defined twice -- as an explicit problem type, then as a
# QuickMDP -- and the two solutions are cross-checked at the end.
#
# Requires POMDPs, POMDPTools, and QuickPOMDPs besides ContinuousDPs and
# its ecosystem (QuantEcon, BasisMatrices). ContinuousDPs and POMDPs
# both export `solve` and `simulate`: with both loaded, qualify the
# calls.
using ContinuousDPs
using BasisMatrices: Basis, ChebParams, LinParams
using QuantEcon: MarkovChain
using POMDPs, POMDPTools, QuickPOMDPs
using Random: MersenneTwister
using Statistics: mean

#= The model as an explicit problem type =#

# Model specification as a subtype of POMDPs.MDP{S,A}
struct Household{TZ<:MarkovChain,TU} <:
        POMDPs.MDP{Tuple{Float64,Float64},Float64}
    r::Float64
    w::Float64
    sigma::Float64
    beta::Float64
    z_chain::TZ
    a_min::Float64
    a_max::Float64
    u::TU
end

function Household(; r = 0.01,
                   w = 1.0,
                   sigma = 1.0,
                   beta = 0.96,
                   z_chain = MarkovChain([0.9 0.1; 0.1 0.9], [0.1; 1.0]),
                   a_min = 1e-10,
                   a_max = 18.0,
                   u = sigma == 1 ? x -> log(x) :
                       x -> (x^(1 - sigma) - 1) / (1 - sigma))
    return Household(r, w, sigma, beta, z_chain, a_min, a_max, u)
end

# The POMDPs.jl interface. Next-period assets are chosen from an
# interval: the budget constraint c > 0 gives the state-dependent upper
# bound (kept away from zero consumption by a small margin).
POMDPs.actions(am::Household, (a, z)::Tuple) =
    ActionInterval(am.a_min,
                   min(am.a_max, am.w * z + (1 + am.r) * a - 1e-8))
POMDPs.reward(am::Household, (a, z)::Tuple, a_new) =
    am.u(am.w * z + (1 + am.r) * a - a_new)
POMDPs.transition(am::Household, (a, z)::Tuple, a_new) =
    SparseCat([(a_new, z_new) for z_new in am.z_chain.state_values],
              am.z_chain.p[findfirst(==(z), am.z_chain.state_values), :])
POMDPs.discount(am::Household) = am.beta

am = Household(; a_max = 20.0, r = 0.03, w = 0.956)

# Collocation basis: Chebyshev in assets (continuous), piecewise linear
# in the productivity state with nodes exactly at its two values (so
# the interpolation in z is exact)
z_vals = am.z_chain.state_values
basis = Basis(ChebParams(30, am.a_min, am.a_max),
              LinParams(collect(z_vals), 0))
policy = POMDPs.solve(CollocationSolver(basis), am; verbose=0)
println("converged: ", policy.res.converged)

# Savings policy a' = a_star(a, z)
a_star(a, z) = action(policy, (a, z))
a_grid = range(am.a_min, am.a_max, length = 5)
println("a_star on a grid (rows: a; columns: z = ", z_vals, "):")
display([a_star(a, z) for a in a_grid, z in z_vals])

# Aggregate capital: the ergodic mean of assets along a long simulated
# path (the continuous-state counterpart of the stationary distribution
# of the controlled Markov chain in the DiscreteDP version)
hist = POMDPs.simulate(HistoryRecorder(max_steps = 100_000,
                                       rng = MersenneTwister(42)),
                       am, policy, (1.0, 1.0))
K = mean(a for (a, z) in state_hist(hist))
println("K = ", K)

#= The same model as a QuickMDP =#

function aiyagari_mdp(; r = 0.01, w = 1.0, sigma = 1.0, beta = 0.96,
                      z_chain = MarkovChain([0.9 0.1; 0.1 0.9], [0.1; 1.0]),
                      a_min = 1e-10, a_max = 18.0,
                      u = sigma == 1 ? x -> log(x) :
                          x -> (x^(1 - sigma) - 1) / (1 - sigma))
    z_vals = z_chain.state_values
    P = z_chain.p
    return QuickMDP(
        statetype = Tuple{Float64,Float64},
        actiontype = Float64,
        actions = ((a, z),) ->
            ActionInterval(a_min, min(a_max, w * z + (1 + r) * a - 1e-8)),
        reward = ((a, z), a_new) -> u(w * z + (1 + r) * a - a_new),
        transition = ((a, z), a_new) ->
            SparseCat([(a_new, z_new) for z_new in z_vals],
                      P[findfirst(==(z), z_vals), :]),
        discount = beta,
    )
end

mq = aiyagari_mdp(; a_max = 20.0, r = 0.03, w = 0.956)
policy_quick = POMDPs.solve(CollocationSolver(basis), mq; verbose=0)
println("QuickMDP solution identical: ",
        policy_quick.res.C == policy.res.C)
