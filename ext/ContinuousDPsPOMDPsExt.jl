# POMDPs.jl interface extension: activated when both POMDPs and
# POMDPTools are loaded.
#
# The headline is the SOLVER direction: `POMDPs.solve(::CollocationSolver,
# m)` solves any explicit-finite POMDPs.jl MDP (finite actions, explicit
# transition distributions, continuous states covered by the solver's
# basis) by the collocation method, via a transition kernel wrapping the
# model. The model direction (`as_mdp`, wrapping a `ContinuousDP` as a
# `POMDPs.MDP`) is internal: it serves as round-trip test infrastructure
# and its public naming is deferred (the eventual generic belongs to
# QuantEcon.jl).
module ContinuousDPsPOMDPsExt

using ContinuousDPs
using ContinuousDPs: CollocationSolver, ContinuousActions, DiscreteActions,
    Interp, ValueFunction, PolicyFunction,
    _action_dim, _policy_eltype, _row,
    _TransitionKernel, _branch_sum, _foreach_branch, _draw_next_state
import POMDPs
using POMDPTools: SparseCat, Deterministic, weighted_iterator
using Random: AbstractRNG

#= Solver direction: CollocationSolver consumes explicit-finite models =#

# Transition kernel wrapping a POMDPs model: branches enumerated from the
# model's explicit transition distribution at each (s, x). States cross
# the boundary as the core's indexable coordinate points (scalars or
# node-row views); the model's next states must likewise be indexable
# (scalars, tuples, or static vectors).
struct _POMDPKernel{TM<:POMDPs.MDP} <: _TransitionKernel
    m::TM
end

function ContinuousDPs._branch_sum(f::F, ker::_POMDPKernel, s, x,
                                   args...) where {F}
    acc = 0.0
    for (sp, w) in weighted_iterator(POMDPs.transition(ker.m, s, x))
        acc += f(sp, w, args...)
    end
    return acc
end

function ContinuousDPs._foreach_branch(f::F, ker::_POMDPKernel, s, x,
                                       args...) where {F}
    for (sp, w) in weighted_iterator(POMDPs.transition(ker.m, s, x))
        f(sp, w, args...)
    end
    return nothing
end

ContinuousDPs._draw_next_state(rng::AbstractRNG, ker::_POMDPKernel, s, x) =
    rand(rng, POMDPs.transition(ker.m, s, x))

# Reward wrappers: infeasible pairs (x outside actions(m, s)) get -Inf
# per the DiscreteActions convention, and the model's transition/reward
# are never evaluated there. The reward arity is chosen once at solve
# time: the direct r(m, s, x) form when the model defines it, otherwise
# the expected form over the branches (a performance choice: the
# expected form costs one reward call per branch per evaluation).
struct _DirectReward{TM<:POMDPs.MDP}
    m::TM
end
(fr::_DirectReward)(s, x) =
    x in POMDPs.actions(fr.m, s) ? POMDPs.reward(fr.m, s, x) : -Inf

_expected_reward_payload(sp, w, m, s, x) = w * POMDPs.reward(m, s, x, sp)

struct _ExpectedReward{TM<:POMDPs.MDP,TK<:_POMDPKernel}
    m::TM
    ker::TK
end
(fr::_ExpectedReward)(s, x) =
    x in POMDPs.actions(fr.m, s) ?
        _branch_sum(_expected_reward_payload, fr.ker, s, x, fr.m, s, x) :
        -Inf

"""
    POMDPs.solve(solver::CollocationSolver, m::POMDPs.MDP; kwargs...)

Solve an explicit-finite POMDPs.jl MDP by the Bellman equation
collocation method and return a `CollocationPolicy`.

Requirements on `m` (checked with informative errors): a finite action
set `actions(m)` (state-dependent restriction via `actions(m, s)` is
supported and mapped to infeasibility, with at least one feasible action
at every collocation node); an explicit transition distribution
(`SparseCat`, `Deterministic`, ... — anything supporting
`POMDPTools.weighted_iterator`); no terminal state at any collocation
node (not supported in this version); rewards as `reward(m, s, x)` or
`reward(m, s, x, sp)`. The state space is continuous with the domain and
dimension given by the solver's basis; states are passed to the model as
indexable coordinate points, and next states must be indexable likewise.
The transition distribution must keep the next states within the basis
domain. Keyword arguments are forwarded to the native `solve`.
"""
function POMDPs.solve(solver::CollocationSolver, m::POMDPs.MDP; kwargs...)
    acts = try
        collect(POMDPs.actions(m))
    catch
        throw(ArgumentError(
            "the collocation solver requires an explicit finite action " *
            "set: `actions(m)` must return a finite collection"))
    end
    isempty(acts) && throw(ArgumentError("`actions(m)` is empty"))

    S = Interp(solver.basis).S
    n = size(S, 1)
    for i in 1:n
        s = _row(S, i)
        POMDPs.isterminal(m, s) && throw(ArgumentError(
            "terminal states are not supported by the collocation " *
            "solver in this version (state $s at collocation node $i " *
            "is terminal)"))
        any(x -> x in POMDPs.actions(m, s), acts) || throw(ArgumentError(
            "no feasible action at collocation node $i (state $s): " *
            "every node needs at least one action in `actions(m, s)`"))
    end

    ker = _POMDPKernel(m)
    # Reward arity by probe call (hasmethod is unreliable: wrappers like
    # QuickMDP define both arities and dispatch to the stored function).
    # Misclassification is safe: the expected form is correct for either
    # arity through POMDPs' reward(m,s,a,sp) = reward(m,s,a) fallback,
    # only costlier.
    s1 = _row(S, 1)
    a1 = something(findfirst(x -> x in POMDPs.actions(m, s1), acts), 1)
    f = try
        POMDPs.reward(m, s1, acts[a1])
        _DirectReward(m)
    catch
        _ExpectedReward(m, ker)
    end
    cdp = ContinuousDP(f=f, g=nothing, discount=POMDPs.discount(m),
                       shocks=Float64[], weights=ker,
                       actions=DiscreteActions(acts))
    res = solve(cdp, solver; kwargs...)
    return CollocationPolicy(m, res, ValueFunction(res),
                             PolicyFunction(res))
end

"""
    CollocationPolicy <: POMDPs.Policy

Policy returned by `POMDPs.solve(solver::CollocationSolver, m)`.
`action(policy, s)` evaluates the computed policy (exact greedy
recomputation for discrete actions, piecewise-linear interpolation
clamped into the action bounds for continuous ones); `value(policy, s)`
evaluates the fitted value function. The full `CDPSolveResult` is
available as `policy.res` (residuals, `set_eval_nodes!`, `simulate`).

Not thread-safe: use one policy instance per thread (the underlying
evaluation caches are single-threaded).
"""
struct CollocationPolicy{TM,TR,TV,TP} <: POMDPs.Policy
    m::TM
    res::TR
    vf::TV
    pf::TP
end

POMDPs.action(policy::CollocationPolicy, s) = policy.pf(s)
POMDPs.value(policy::CollocationPolicy, s) = policy.vf(s)

#= Model direction (internal): a ContinuousDP viewed as a POMDPs.MDP =#

"""
    CDPMDP{S,A} <: POMDPs.MDP{S,A}

A `ContinuousDP` viewed as a `POMDPs.MDP`; internal (construct with the
ext-local [`as_mdp`](@ref)). The state type `S` is `Float64` for a
scalar state or `NTuple{N,Float64}` for an `N`-dimensional one; the
action type `A` is `Float64` for a continuous action space, the action
value type for a discrete one. The wrapped problem's primitives are
shared, not copied.
"""
struct CDPMDP{S,A,TCDP<:ContinuousDP,TI} <: POMDPs.MDP{S,A}
    cdp::TCDP
    initialstate::TI
end

_statedim(::CDPMDP{Float64}) = 1
_statedim(::CDPMDP{NTuple{N,Float64}}) where {N} = N

_to_state(::Type{Float64}, sp) = Float64(sp)
_to_state(::Type{NTuple{N,Float64}}, sp) where {N} =
    ntuple(d -> Float64(sp[d]), Val(N))

"""
    as_mdp(cdp::ContinuousDP; initialstate=nothing, statedim=1)

Wrap `cdp` as a `CDPMDP` (internal; the public naming of this operation
is deferred). `statedim` declares the state dimension, which a
primitives-only `ContinuousDP` does not carry; the state type is
`Float64` for `statedim == 1` and `NTuple{statedim,Float64}` otherwise.
`initialstate` may be a state, a number (scalar state), or a POMDPs
distribution; if omitted, `POMDPs.initialstate` throws an informative
error. Requires a fixed weights vector (callable weights are not
supported by the model direction) and a scalar or discrete action space.
"""
function as_mdp(cdp::ContinuousDP; initialstate=nothing, statedim::Int=1)
    statedim >= 1 || throw(ArgumentError("statedim must be positive"))
    cdp.weights isa AbstractVector || throw(ArgumentError(
        "as_mdp requires a fixed weights vector (callable weights are " *
        "not supported by the model direction)"))
    a = cdp.actions
    a isa ContinuousActions && _action_dim(a) > 1 && throw(ArgumentError(
        "as_mdp supports scalar continuous actions only (the action " *
        "space is $(_action_dim(a))-dimensional)"))
    S = statedim == 1 ? Float64 : NTuple{statedim,Float64}
    init = if initialstate === nothing
        nothing
    elseif initialstate isa Real && statedim == 1
        Deterministic(Float64(initialstate))
    elseif initialstate isa Union{Tuple,AbstractVector}
        Deterministic(_to_state(S, initialstate))
    else
        initialstate  # a POMDPs distribution
    end
    A = _policy_eltype(a)
    return CDPMDP{S,A,typeof(cdp),typeof(init)}(cdp, init)
end

function POMDPs.transition(m::CDPMDP{S}, s, x) where {S}
    cdp = m.cdp
    K = size(cdp.shocks, 1)
    sps = [_to_state(S, cdp.g(s, x, _row(cdp.shocks, j))) for j in 1:K]
    return SparseCat(sps, cdp.weights)
end

POMDPs.reward(m::CDPMDP, s, x) = m.cdp.f(s, x)
POMDPs.discount(m::CDPMDP) = m.cdp.discount
POMDPs.isterminal(m::CDPMDP, s) = false

"""
    CDPActionInterval

Closed interval of feasible actions `[lo, hi]` at a given state, returned
by `POMDPs.actions(m::CDPMDP, s)` for a continuous action space. Supports
`rand`, `minimum`, `maximum`, and `in`.
"""
struct CDPActionInterval
    lo::Float64
    hi::Float64
end

Base.rand(rng::AbstractRNG, itv::CDPActionInterval) =
    itv.lo + (itv.hi - itv.lo) * rand(rng)
Base.minimum(itv::CDPActionInterval) = itv.lo
Base.maximum(itv::CDPActionInterval) = itv.hi
Base.in(x, itv::CDPActionInterval) = itv.lo <= x <= itv.hi

POMDPs.actions(m::CDPMDP, s) = _actions(m.cdp.actions, s)
_actions(a::DiscreteActions, s) = a.vals
_actions(a::ContinuousActions, s) =
    CDPActionInterval(Float64(a.x_lb(s)), Float64(a.x_ub(s)))

function POMDPs.actions(m::CDPMDP)
    a = m.cdp.actions
    a isa DiscreteActions || throw(ArgumentError(
        "the action set of a continuous-action CDPMDP is state-dependent: " *
        "use `actions(m, s)`"))
    return a.vals
end

function POMDPs.initialstate(m::CDPMDP)
    m.initialstate === nothing && throw(ArgumentError(
        "no initial state distribution was supplied: pass `initialstate` " *
        "to `as_mdp`, or give simulators an explicit start state"))
    return m.initialstate
end

function POMDPs.solve(solver::CollocationSolver, m::CDPMDP; kwargs...)
    nd = ndims(solver.basis)
    nd == _statedim(m) || throw(ArgumentError(
        "the solver basis has $nd dimension(s) but the CDPMDP state is " *
        "$(_statedim(m))-dimensional"))
    res = solve(m.cdp, solver; kwargs...)
    return CollocationPolicy(m, res, ValueFunction(res),
                             PolicyFunction(res))
end

end # module
