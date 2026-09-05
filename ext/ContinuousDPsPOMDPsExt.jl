# POMDPs.jl interface extension: activated when both POMDPs and
# POMDPTools are loaded.
#
# The headline is MODEL IMPORT: `POMDPs.solve(::CollocationSolver, m)`
# solves any explicit-finite POMDPs.jl MDP (finite actions, explicit
# transition distributions, continuous states covered by the solver's
# basis) by the collocation method, via a transition kernel wrapping the
# model. Model export (`as_mdp`, wrapping a `ContinuousDP` as a
# `POMDPs.MDP`) is internal: it serves as round-trip test infrastructure
# and its public naming is deferred (the eventual generic belongs to
# QuantEcon.jl).
module ContinuousDPsPOMDPsExt

using ContinuousDPs
using ContinuousDPs: CollocationSolver, ContinuousActions, DiscreteActions,
    ActionInterval, Interp, ValueFunction, PolicyFunction,
    _action_dim, _policy_eltype, _row,
    _TransitionKernel, _branch_sum, _foreach_branch, _draw_next_state
import POMDPs
using POMDPTools: SparseCat, Deterministic, weighted_iterator
using Random: AbstractRNG

#= Model import: the collocation solver consumes explicit-finite models =#

# Solver-to-model state conversion, fixed once at solve time by probing
# the first collocation node: the core's coordinate points (scalars in
# 1-D, node-row views in N-D) pass through when they already match
# `statetype(m)`; `Tuple` state types get the natural elementwise
# conversion; anything else is routed through `POMDPs.convert_s` when
# the model provides a method. With no applicable conversion,
# coordinates pass through under the documented indexable-points
# contract. Every converter is a no-op on an already-converted state,
# so nested application is safe.
struct _TupleState{S} end
(::_TupleState{S})(s) where {S} = s isa S ? s : convert(S, Tuple(s))

struct _ConvertedState{S,TM<:POMDPs.MDP}
    m::TM
end
(c::_ConvertedState{S})(s) where {S} =
    s isa S ? s : POMDPs.convert_s(S, s, c.m)

function _state_converter(m::POMDPs.MDP, s_probe)
    S = POMDPs.statetype(m)
    s_probe isa S && return identity
    S <: Tuple && return _TupleState{S}()
    conv = _ConvertedState{S,typeof(m)}(m)
    try
        conv(s_probe)
        return conv
    catch err
        err isa InterruptException && rethrow()
        return identity
    end
end

# Transition kernel wrapping a POMDPs model: branches enumerated from the
# model's explicit transition distribution at each (s, x). States cross
# the boundary through the `to_state` converter above; the model's next
# states must be indexable (scalars, tuples, or static vectors).
struct _POMDPKernel{TM<:POMDPs.MDP,TC} <: _TransitionKernel
    m::TM
    to_state::TC
end

function ContinuousDPs._branch_sum(f::F, ker::_POMDPKernel, s, x,
                                   args...) where {F}
    ms = ker.to_state(s)
    acc = 0.0
    for (sp, w) in weighted_iterator(POMDPs.transition(ker.m, ms, x))
        acc += f(sp, w, args...)
    end
    return acc
end

function ContinuousDPs._foreach_branch(f::F, ker::_POMDPKernel, s, x,
                                       args...) where {F}
    ms = ker.to_state(s)
    for (sp, w) in weighted_iterator(POMDPs.transition(ker.m, ms, x))
        f(sp, w, args...)
    end
    return nothing
end

ContinuousDPs._draw_next_state(rng::AbstractRNG, ker::_POMDPKernel, s, x) =
    rand(rng, POMDPs.transition(ker.m, ker.to_state(s), x))

# Reward wrappers: infeasible pairs (x outside actions(m, s)) get -Inf
# per the DiscreteActions convention, and the model's transition/reward
# are never evaluated there. The reward arity is chosen once at solve
# time: the direct r(m, s, x) form when the model defines it, otherwise
# the expected form over the branches (a performance choice: the
# expected form costs one reward call per branch per evaluation).
struct _DirectReward{TM<:POMDPs.MDP,TC}
    m::TM
    to_state::TC
end
function (fr::_DirectReward)(s, x)
    ms = fr.to_state(s)
    return x in POMDPs.actions(fr.m, ms) ? POMDPs.reward(fr.m, ms, x) : -Inf
end

_expected_reward_payload(sp, w, m, s, x) = w * POMDPs.reward(m, s, x, sp)

struct _ExpectedReward{TM<:POMDPs.MDP,TK<:_POMDPKernel}
    m::TM
    ker::TK
end
function (fr::_ExpectedReward)(s, x)
    ms = fr.ker.to_state(s)
    return x in POMDPs.actions(fr.m, ms) ?
        _branch_sum(_expected_reward_payload, fr.ker, ms, x, fr.m, ms, x) :
        -Inf
end

# Action bounds of a continuous action space declared through
# `actions(m, s)::ActionInterval`, as callables of the solver's coordinates
struct _IntervalBound{W,TM<:POMDPs.MDP,TC}
    m::TM
    to_state::TC
end
(b::_IntervalBound{:lo})(s) = minimum(POMDPs.actions(b.m, b.to_state(s)))
(b::_IntervalBound{:hi})(s) = maximum(POMDPs.actions(b.m, b.to_state(s)))

"""
    POMDPs.solve(solver::CollocationSolver, m::POMDPs.MDP; kwargs...)

Solve an explicit-finite POMDPs.jl MDP by the Bellman equation
collocation method and return a `CollocationPolicy`.

Requirements on `m` (checked with informative errors where feasible):
an action space given either as a finite action set `actions(m)`
(state-dependent restriction via `actions(m, s)` is supported and mapped
to infeasibility, with at least one feasible action at every collocation
node; `actions(m, s)` must be a subset of `actions(m)` — actions outside
the global set are not seen by the solver), or as a scalar continuous
action space declared by `actions(m, s)` returning an
[`ActionInterval`](@ref) (the inner maximization then uses the
derivative-free solver, and the model's action type must accept
`Float64`); an explicit transition distribution (`SparseCat`,
`Deterministic`, ... — anything supporting
`POMDPTools.weighted_iterator`); no terminal states: `isterminal(m, s)`
must be `false` on the entire basis domain (collocation nodes are
checked; off-node states are the model's responsibility); rewards as
`reward(m, s, x)` or `reward(m, s, x, sp)`. The state space is
continuous with the domain and dimension given by the solver's basis.
States are passed to the model as `statetype(m)` when a conversion from
the solver's coordinates applies (exact match, a `Tuple` state type, or
a `POMDPs.convert_s` method from the coordinate form), and as indexable
coordinate points otherwise; next states must be indexable (scalars,
tuples, or static vectors) and must stay within the basis domain.
Keyword arguments are forwarded to the native `solve`.
"""
function POMDPs.solve(solver::CollocationSolver, m::POMDPs.MDP; kwargs...)
    S = Interp(solver.basis).S
    n = size(S, 1)
    to_state = _state_converter(m, _row(S, 1))
    s1 = to_state(_row(S, 1))

    # Action space: an ActionInterval from actions(m, s) declares a
    # scalar continuous action space; otherwise actions(m) must be a
    # finite collection (with actions(m, s) restricting it per state)
    acts1 = POMDPs.actions(m, s1)
    continuous = acts1 isa ActionInterval
    if continuous
        Float64 <: POMDPs.actiontype(m) || throw(ArgumentError(
            "a continuous action space is solved and returned as " *
            "Float64 actions: the model's action type must accept " *
            "Float64 (got actiontype $(POMDPs.actiontype(m)))"))
        actions = ContinuousActions(
            _IntervalBound{:lo,typeof(m),typeof(to_state)}(m, to_state),
            _IntervalBound{:hi,typeof(m),typeof(to_state)}(m, to_state))
        a_probe = 0.5 * (minimum(acts1) + maximum(acts1))
    else
        acts = try
            collect(POMDPs.actions(m))
        catch err
            err isa InterruptException && rethrow()
            throw(ArgumentError(
                "the collocation solver requires an explicit finite " *
                "action set: `actions(m)` must return a finite " *
                "collection, or `actions(m, s)` an `ActionInterval` " *
                "(collecting it failed with $(sprint(showerror, err)))"))
        end
        isempty(acts) && throw(ArgumentError("`actions(m)` is empty"))
        actions = DiscreteActions(acts)
        a_probe = acts[something(findfirst(x -> x in acts1, acts), 1)]
    end

    for i in 1:n
        s = to_state(_row(S, i))
        POMDPs.isterminal(m, s) && throw(ArgumentError(
            "terminal states are not supported by the collocation " *
            "solver in this version (state $s at collocation node $i " *
            "is terminal)"))
        continuous && continue
        any(x -> x in POMDPs.actions(m, s), acts) || throw(ArgumentError(
            "no feasible action at collocation node $i (state $s): " *
            "every node needs at least one action in `actions(m, s)`"))
    end

    ker = _POMDPKernel(m, to_state)
    # Reward arity by probe call (hasmethod is unreliable: wrappers like
    # QuickMDP define both arities and dispatch to the stored function).
    # Misclassification is safe: the expected form is correct for either
    # arity through POMDPs' reward(m,s,a,sp) = reward(m,s,a) fallback,
    # only costlier.
    f = try
        POMDPs.reward(m, s1, a_probe)
        _DirectReward(m, to_state)
    catch err
        err isa InterruptException && rethrow()
        _ExpectedReward(m, ker)
    end
    cdp = ContinuousDP(f=f, g=nothing, discount=POMDPs.discount(m),
                       shocks=Float64[], weights=ker, actions=actions)
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
available as `policy.res` (residuals, `set_eval_nodes!`, `simulate`);
after `set_eval_nodes!(policy.res, ...)`, `action` evaluates the policy
on the new evaluation nodes.

Not thread-safe: use one policy instance per thread (the underlying
evaluation caches are single-threaded).
"""
mutable struct CollocationPolicy{TM,TR,TV,TP,TX,TN} <: POMDPs.Policy
    m::TM
    res::TR
    vf::TV
    pf::TP
    # The result fields the policy functor was built from: a continuous
    # policy interpolates `res.X` over `res.eval_nodes_coord`, both of
    # which `set_eval_nodes!` rebinds
    pf_X::TX
    pf_nodes::TN
end

CollocationPolicy(m, res, vf, pf) =
    CollocationPolicy(m, res, vf, pf, res.X, res.eval_nodes_coord)

# Rebuild the policy functor if the evaluation nodes changed since it was
# built (a pointer comparison per call otherwise)
function _policy_function(policy::CollocationPolicy)
    res = policy.res
    if res.X !== policy.pf_X || res.eval_nodes_coord !== policy.pf_nodes
        policy.pf = PolicyFunction(res)
        policy.pf_X = res.X
        policy.pf_nodes = res.eval_nodes_coord
    end
    return policy.pf
end

POMDPs.action(policy::CollocationPolicy, s) = _policy_function(policy)(s)
POMDPs.value(policy::CollocationPolicy, s) = policy.vf(s)

#= Model export (internal): a ContinuousDP viewed as a POMDPs.MDP =#

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
error. Requires a fixed weights vector forming a probability
distribution (callable or sub-stochastic weights are not supported by
model export) and a scalar or discrete action space.
"""
function as_mdp(cdp::ContinuousDP; initialstate=nothing, statedim::Int=1)
    statedim >= 1 || throw(ArgumentError("statedim must be positive"))
    cdp.weights isa AbstractVector || throw(ArgumentError(
        "as_mdp requires a fixed weights vector (callable weights are " *
        "not supported by model export)"))
    # A POMDPs distribution needs a proper probability vector: the
    # sub-stochastic or unnormalized weights the native solver permits
    # would be read raw by `weighted_iterator` but rescaled by `rand`
    w = cdp.weights
    (all(wj -> isfinite(wj) && wj >= 0, w) &&
     isapprox(sum(w), 1; atol=1e-8)) || throw(ArgumentError(
        "as_mdp requires the weights to form a probability vector " *
        "(finite, nonnegative, summing to one)"))
    a = cdp.actions
    a isa ContinuousActions && _action_dim(a) > 1 && throw(ArgumentError(
        "as_mdp supports scalar continuous actions only (the action " *
        "space is $(_action_dim(a))-dimensional)"))
    S = statedim == 1 ? Float64 : NTuple{statedim,Float64}
    init = if initialstate === nothing
        nothing
    elseif initialstate isa Real
        statedim == 1 || throw(ArgumentError(
            "a scalar initial state requires statedim == 1"))
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

# For a continuous action space the feasible set at a state is an
# ActionInterval (the same representation model import recognizes)
POMDPs.actions(m::CDPMDP, s) = _actions(m.cdp.actions, s)
_actions(a::DiscreteActions, s) = a.vals
_actions(a::ContinuousActions, s) =
    ActionInterval(Float64(a.x_lb(s)), Float64(a.x_ub(s)))

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
