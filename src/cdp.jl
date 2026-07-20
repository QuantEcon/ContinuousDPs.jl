#=
Tools for representing and solving dynamic programs with continuous states.

Implement the Bellman equation collocation method as described in Miranda and
Fackler (2002), Chapter 9.

References
----------
* M. J. Miranda and P. L. Fackler, Applied Computational Economics and Finance,
  MIT Press, 2002.

=#
using BasisMatrices
import Optim
using NLSolversBase: only_fg!
using FiniteDiff
using SparseArrays: SparseMatrixCSC, sparse
import QuantEcon.ScalarOrArray


#= Types and constructors =#

"""
    Interp{N,TB,TS,TM,TL}

Type representing an interpolation scheme on an `N`-dimensional domain.

# Fields

- `basis::TB<:Basis{N}`: Object that contains the interpolation basis
  information.
- `S::TS<:VecOrMat`: Vector or Matrix that contains interpolation nodes
  (collocation points).
- `Scoord::NTuple{N,Vector{Float64}}`: Coordinate vectors of the interpolation
  nodes along each dimension.
- `length::Int`: Total number of interpolation nodes on the tensor grid.
- `size::NTuple{N,Int}`: Number of interpolation nodes along each dimension.
- `lb::NTuple{N,Float64}`: Lower bounds of the domain.
- `ub::NTuple{N,Float64}`: Upper bounds of the domain.
- `Phi::TM<:AbstractMatrix`: Basis matrix evaluated at the interpolation nodes.
- `Phi_lu::TL<:Factorization`: LU factorization of `Phi`.
"""
struct Interp{N,TB<:Basis{N},TS<:VecOrMat,TM<:AbstractMatrix,TL<:Factorization}
    basis::TB
    S::TS
    Scoord::NTuple{N,Vector{Float64}}
    length::Int
    size::NTuple{N,Int}
    lb::NTuple{N,Float64}
    ub::NTuple{N,Float64}
    Phi::TM
    Phi_lu::TL
end

"""
    Interp(basis)

Construct an `Interp` from a `Basis`.

# Arguments

- `basis::Basis`: Object that contains the interpolation basis information.
"""
function Interp(basis::Basis{N}) where {N}
    S, Scoord = nodes(basis)
    grid_length = length(basis)
    grid_size = size(basis)
    grid_lb, grid_ub = min(basis), max(basis)
    Phi = BasisMatrix(basis, Expanded(), S).vals[1]
    Phi_lu = lu(Phi)
    interp = Interp(basis, S, Scoord, grid_length, grid_size, grid_lb, grid_ub,
                    Phi, Phi_lu)
end


#= Action spaces =#

"""
    ActionSpace

Abstract type for action spaces of a `ContinuousDP`. Concrete subtypes:
[`ContinuousActions`](@ref), [`DiscreteActions`](@ref).
"""
abstract type ActionSpace end

"""
    ContinuousActions{M,Tlb,Tub}

Continuous action space: an `M`-dimensional box `[x_lb(s), x_ub(s)]` that may
depend on the state. Construct with `ContinuousActions(x_lb, x_ub)` for
`M == 1` (with `x_lb(s)`, `x_ub(s)` returning scalars) or
`ContinuousActions{M}(x_lb, x_ub)` for `M > 1` (with the bound functions
returning length-`M` tuples or vectors).

For `M > 1`, actions are passed to the reward and transition functions as
length-`M` collections indexable by `x[1], ..., x[M]` (tuples or views), and
policy functions are stored as `n x M` matrices (one row per state node).

# Fields

- `x_lb::Tlb`: Lower bound of the action as a function of the state.
- `x_ub::Tub`: Upper bound of the action as a function of the state.
"""
struct ContinuousActions{M,Tlb,Tub} <: ActionSpace
    x_lb::Tlb
    x_ub::Tub

    function ContinuousActions{M,Tlb,Tub}(x_lb, x_ub) where {M,Tlb,Tub}
        (M isa Int && M >= 1) || throw(ArgumentError(
            "the action dimension M must be a positive integer"))
        return new{M,Tlb,Tub}(x_lb, x_ub)
    end
end

ContinuousActions(x_lb, x_ub) =
    ContinuousActions{1,typeof(x_lb),typeof(x_ub)}(x_lb, x_ub)

ContinuousActions{M}(x_lb, x_ub) where {M} =
    ContinuousActions{M,typeof(x_lb),typeof(x_ub)}(x_lb, x_ub)

"""
    DiscreteActions{TA}

Discrete (finite) action space, given by a vector of action values of any
homogeneous type `TA` — numbers, tuples, labels, etc.; the values are passed
opaquely to the reward and transition functions. Internally the solvers work
with the *indices* into `vals`, and the solution exposes both the values
(`res.X`) and the indices (`res.X_ind`), following the
`MarkovChain`/`state_values` convention of QuantEcon.

Infeasible state-action pairs are expressed by the reward function returning
`-Inf`; the enumeration over actions then skips the transition evaluation
for that candidate. A well-posed model should have at least one feasible
action at every state the solver evaluates. If every action is infeasible
at some state, the first action is retained as a fallback, and subsequent
policy evaluation may call `g` for that pair — `g` should tolerate such
calls.

# Fields

- `vals::Vector{TA}`: Action values.
"""
struct DiscreteActions{TA} <: ActionSpace
    vals::Vector{TA}

    function DiscreteActions(vals::AbstractVector{TA}) where TA
        isempty(vals) && throw(ArgumentError("action set must be nonempty"))
        return new{TA}(collect(vals))
    end
end

Base.length(a::DiscreteActions) = length(a.vals)

# Element type of the policy-function container for each action space
_policy_eltype(::ContinuousActions) = Float64
_policy_eltype(::DiscreteActions{TA}) where TA = TA

# Policy-function containers: a length-n vector, except an n x M matrix for
# multi-dimensional continuous actions (rows are action points, matching the
# convention for state nodes S). Entries are initialized to NaN for
# continuous actions (NaN marks "no warm start").
_empty_policy(::ContinuousActions{1}) = Float64[]
_empty_policy(::ContinuousActions{M}) where M = Matrix{Float64}(undef, 0, M)
_empty_policy(::DiscreteActions{TA}) where TA = TA[]

_policy_buffer(::ContinuousActions{1}, n::Int) = fill(NaN, n)
_policy_buffer(::ContinuousActions{M}, n::Int) where M = fill(NaN, n, M)
_policy_buffer(::DiscreteActions{TA}, n::Int) where TA =
    Vector{TA}(undef, n)

_policy_size_ok(a::ContinuousActions{1}, X, n) = length(X) == n
_policy_size_ok(a::ContinuousActions{M}, X, n) where M = size(X) == (n, M)
_policy_size_ok(a::DiscreteActions, X, n) = length(X) == n

# Action dimension as a compile-time constant
_action_dim(::ContinuousActions{M}) where M = M


"""
    ContinuousDP{Tf,Tg,TR,TA}

Type representing a continuous-state dynamic program. A `ContinuousDP` holds
the primitives of the problem; the interpolation scheme used to solve it is
supplied separately through a solver object (see [`CollocationSolver`](@ref)
and [`LQASolver`](@ref)).

# Fields

- `f::Tf`: Reward function `f(s, x)`.
- `g::Tg`: State transition function `g(s, x, e)`.
- `discount::Float64`: Discount factor.
- `shocks::TR<:AbstractVecOrMat`: Discretized shock nodes.
- `weights::TW`: Probability weights for the shock nodes: a
  `Vector{Float64}` with one weight per shock node
  (`length(weights) == size(shocks, 1)`) for a fixed distribution, or a
  callable `weights(s)` / `weights(s, x)` for a state- or
  state-action-dependent distribution (see the constructor).
- `actions::TA<:ActionSpace`: Action space (see [`ContinuousActions`](@ref)
  and [`DiscreteActions`](@ref)).
"""
struct ContinuousDP{Tf,Tg,TR<:AbstractVecOrMat,TW,TA<:ActionSpace}
    f::Tf
    g::Tg
    discount::Float64
    shocks::TR
    weights::TW
    actions::TA

    # The explicit inner constructor suppresses the implicit outer one,
    # which would otherwise be more specific than (and shadow) the
    # `discount::Real` positional constructor below whenever `discount`
    # is already a Float64 (which runs `_process_weights`). The
    # fixed-weights length invariant lives here, the boundary every
    # construction path must pass; callable weights are validated at
    # kernel construction instead (see `_build_kernel`).
    function ContinuousDP{Tf,Tg,TR,TW,TA}(
            f, g, discount, shocks, weights, actions
        ) where {Tf,Tg,TR<:AbstractVecOrMat,TW,TA<:ActionSpace}
        if weights isa AbstractVector
            length(weights) == size(shocks, 1) || throw(ArgumentError(
                "`weights` must have one weight per shock node " *
                "($(size(shocks, 1))); got $(length(weights))"))
        end
        return new{Tf,Tg,TR,TW,TA}(f, g, discount, shocks, weights, actions)
    end
end

"""
    ContinuousDP(; f, g, discount, x_lb, x_ub, shocks, weights)
    ContinuousDP(; f, g, discount, actions, shocks, weights)
    ContinuousDP(f, g, discount, shocks, weights, actions)
    ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub)

Constructor for `ContinuousDP`. The problem is specified by its primitives
only; the interpolation basis is supplied separately through the solver
passed to `solve` (see [`CollocationSolver`](@ref) and [`LQASolver`](@ref)).
Give either `actions`, or both `x_lb` and `x_ub` (equivalent to
`actions = ContinuousActions(x_lb, x_ub)`).

# Arguments

- `f`: Reward function `f(s, x)`.
- `g`: State transition function `g(s, x, e)`.
- `discount::Real`: Discount factor.
- `actions::ActionSpace`: Action space; alternatively give `x_lb`, `x_ub`
  (lower and upper bound of the action as functions of the state) for a
  one-dimensional continuous action space.
- `shocks::AbstractVecOrMat`: Discretized shock nodes.
- `weights`: Probability weights for the shock nodes. Either a fixed
  probability vector with one weight per shock node
  (`length(weights) == size(shocks, 1)`), or a callable returning such a
  collection: `weights(s)` for a state-dependent distribution, or
  `weights(s, x)` for a state-action-dependent one (if both arities
  apply, the state-action form is used). A callable returning a `Tuple`
  or a statically-sized vector (e.g. a `StaticArrays.SVector`) keeps the
  solver sweeps allocation-free; returning a freshly allocated `Vector`
  is supported but allocation-lean. Callable weights are validated only
  by a length check at a probe point (skipped if the probe call errors);
  the solve operators do not check that weights sum to one
  (sub-stochastic weights are permitted and act as additional
  discounting). Simulation, by contrast, requires a proper probability
  vector at every visited `(s, x)` — `simulate` throws an
  `ArgumentError` for weights that are negative, non-finite, or do not
  sum to one, since the missing mass has no path-wise interpretation.
  With action-dependent weights the first-order-condition inner solver
  does not apply and `solve` automatically falls back to Brent (see
  [`solve`](@ref)).
"""
function ContinuousDP(f, g, discount::Real,
                      shocks::AbstractVecOrMat, weights,
                      actions::ActionSpace)
    w = _process_weights(weights, shocks)
    return ContinuousDP{typeof(f),typeof(g),typeof(shocks),typeof(w),
                        typeof(actions)}(
        f, g, Float64(discount), shocks, w, actions)
end

# Vector weights convert to the canonical Vector{Float64} (the length
# invariant is enforced by the inner constructor); anything else must at
# least be callable, with full validation at solve entry (workspace
# creation; see _validate_weights)
_process_weights(weights::AbstractVector, shocks) =
    convert(Vector{Float64}, weights)

function _process_weights(weights, shocks)
    isempty(methods(weights)) && throw(ArgumentError(
        "`weights` must be a probability vector or a callable " *
        "`weights(s)` / `weights(s, x)`"))
    return weights
end

ContinuousDP(f, g, discount::Real,
             shocks::AbstractVecOrMat, weights,
             x_lb, x_ub) =
    ContinuousDP(f, g, discount, shocks, weights,
                 ContinuousActions(x_lb, x_ub))

function ContinuousDP(; f, g, discount, shocks, weights,
                      x_lb=nothing, x_ub=nothing, actions=nothing)
    if actions === nothing
        (x_lb !== nothing && x_ub !== nothing) || throw(ArgumentError(
            "give either `actions`, or both `x_lb` and `x_ub`"))
        actions = ContinuousActions(x_lb, x_ub)
    else
        (x_lb === nothing && x_ub === nothing) || throw(ArgumentError(
            "the `x_lb`/`x_ub` keywords cannot be combined with `actions`"))
        actions isa ActionSpace || throw(ArgumentError(
            "`actions` must be an ActionSpace " *
            "(ContinuousActions or DiscreteActions)"))
    end
    return ContinuousDP(f, g, discount, shocks, weights, actions)
end

# TODO: delete these error stubs in a later release (kept for migration
# guidance from the v0.2 API, removed in v0.3). The 7-arg stub is
# load-bearing beyond ergonomics: without it, the removed
# `(..., actions, basis)` form would silently match the primitives
# `(..., x_lb, x_ub)` method and construct a nonsense action space.
ContinuousDP(f, g, discount::Real,
             shocks::AbstractVecOrMat, weights,
             actions_or_x_lb, basis::Basis) =
    throw(ArgumentError(
        "the basis-endowed `ContinuousDP` constructors have been removed: " *
        "construct the problem without `basis` and pass " *
        "`CollocationSolver(basis)` to `solve`"))
ContinuousDP(f, g, discount::Real,
             shocks::AbstractVecOrMat, weights,
             x_lb, x_ub, basis::Basis) =
    throw(ArgumentError(
        "the basis-endowed `ContinuousDP` constructors have been removed: " *
        "construct the problem without `basis` and pass " *
        "`CollocationSolver(basis)` to `solve`"))

"""
    ContinuousDP(cdp::ContinuousDP; f=cdp.f, g=cdp.g, discount=cdp.discount,
                 shocks=cdp.shocks, weights=cdp.weights, actions=cdp.actions,
                 x_lb=nothing, x_ub=nothing)

Construct a copy of `cdp`, optionally replacing selected model components.
The `x_lb`/`x_ub` keywords replace the corresponding bound of a continuous
action space.
"""
function ContinuousDP(cdp::ContinuousDP;
    f = cdp.f,
    g = cdp.g,
    discount = cdp.discount,
    shocks = cdp.shocks,
    weights = cdp.weights,
    actions = cdp.actions,
    x_lb = nothing,
    x_ub = nothing
)
    if x_lb !== nothing || x_ub !== nothing
        actions isa ContinuousActions || throw(ArgumentError(
            "x_lb/x_ub keywords apply only to continuous action spaces"))
        actions = ContinuousActions{_action_dim(actions)}(
            x_lb === nothing ? actions.x_lb : x_lb,
            x_ub === nothing ? actions.x_ub : x_ub
        )
    end
    out = ContinuousDP(f, g, discount, shocks, weights, actions)
    return out
end


# Internal: a ContinuousDP bound to an interpolation scheme. All solver
# internals from the node-sweep level up consume this pair; ContinuousDP
# itself stays primitives-only. The state-local point kernels
# (_s_wise_max! and friends) take the plain ContinuousDP: they are
# interpolation-free by design (they only need a FunEvalCache).
struct _CollocationProblem{N,TCDP<:ContinuousDP,TI<:Interp{N}}
    cdp::TCDP
    interp::TI
end

Base.ndims(::_CollocationProblem{N}) where {N} = N

"""
    CDPSolveResult{Algo,N,TCDP,TI,TE,TX}

Type storing the solution of a continuous-state dynamic program obtained by
algorithm `Algo`.

# Fields

- `cdp::TCDP<:ContinuousDP`: The dynamic program that was solved.
- `interp::TI<:Interp{N}`: The interpolation scheme used by the solver.
- `tol::Float64`: Convergence tolerance used by the solver.
- `max_iter::Int`: Maximum number of iterations allowed.
- `C::Vector{Float64}`: Basis coefficient vector for the fitted value function.
- `converged::Bool`: Whether the algorithm converged.
- `num_iter::Int`: Number of iterations performed.
- `inner_solver::Symbol`: Inner solver used by `solve` (`:foc` or
  `:brent`); also used when re-evaluating the policy (e.g. by
  `set_eval_nodes!`) for multi-dimensional continuous actions.
- `eval_nodes::TE<:VecOrMat`: Nodes at which the solution is evaluated.
  Defaults to `interp.S`.
- `eval_nodes_coord::NTuple{N,Vector{Float64}}`: Coordinate vectors of the
  evaluation nodes along each dimension. Defaults to `interp.Scoord`.
- `V::Vector{Float64}`: Value function evaluated at `eval_nodes`.
- `X::TX<:AbstractVecOrMat`: Policy function (action values) evaluated at
  `eval_nodes` (an `n x M` matrix for `M`-dimensional continuous actions).
- `X_ind::Vector{Int}`: For a discrete action space, the indices into
  `cdp.actions.vals` corresponding to `X`; empty otherwise.
- `resid::Vector{Float64}`: Approximation residuals at `eval_nodes`.
"""
mutable struct CDPSolveResult{Algo<:DPAlgorithm,N,TCDP<:ContinuousDP,
                              TI<:Interp{N},TE<:VecOrMat,
                              TX<:AbstractVecOrMat}
    cdp::TCDP
    interp::TI
    tol::Float64
    max_iter::Int
    C::Vector{Float64}
    converged::Bool
    num_iter::Int
    inner_solver::Symbol
    eval_nodes::TE
    eval_nodes_coord::NTuple{N,Vector{Float64}}
    V::Vector{Float64}
    X::TX
    X_ind::Vector{Int}
    resid::Vector{Float64}

    function CDPSolveResult{Algo}(
            cdp::TCDP, interp::Interp{N}, tol::Float64, max_iter::Integer,
            inner_solver::Symbol=:foc
        ) where {Algo,N,TCDP<:ContinuousDP}
        C = zeros(interp.length)
        converged = false
        num_iter = 0
        eval_nodes = interp.S
        eval_nodes_coord = interp.Scoord
        V = Float64[]
        X = _empty_policy(cdp.actions)
        X_ind = Int[]
        resid = Float64[]
        res = new{Algo,N,TCDP,typeof(interp),typeof(eval_nodes),typeof(X)}(
            cdp, interp, tol, max_iter, C, converged, num_iter, inner_solver,
            eval_nodes, eval_nodes_coord, V, X, X_ind, resid
        )
        return res
    end
end

# Rebuild the internal bound problem from a result
_colloc(res::CDPSolveResult) = _CollocationProblem(res.cdp, res.interp)

# Derivative caches for gradient-based evaluation sweeps, or nothing when
# the basis does not support them (then the derivative-free path is used)
function _eval_dfecs(interp::Interp{N}) where N
    basis = interp.basis
    all(d -> _foc_suitable(basis.params[d]), 1:N) || return nothing
    return ntuple(d -> DerivFunEvalCache(
               basis, ntuple(i -> Int(i == d), Val(N))), Val(N))
end

Base.ndims(::CDPSolveResult{Algo,N}) where {Algo,N} = N

"""
    evaluate!(res[, fec])

Evaluate the value function and the policy function at the evaluation nodes.

The result arrays `res.V`, `res.X`, and `res.resid` are updated in place
when their lengths already match the number of evaluation nodes, and
reallocated otherwise.

# Arguments

- `res::CDPSolveResult`: Solution object to update in place.
- `fec::FunEvalCache`: Workspace for point evaluation of the value function.
  Constructed internally if not given.
"""
function evaluate!(res::CDPSolveResult, fec::FunEvalCache)
    cdp, C, s_nodes = res.cdp, res.C, res.eval_nodes
    a = cdp.actions
    ker = _build_kernel(_colloc(res))
    n = size(s_nodes, 1)
    length(res.V) == n || (res.V = Vector{Float64}(undef, n))
    _policy_size_ok(a, res.X, n) || (res.X = _policy_buffer(a, n))
    length(res.resid) == n || (res.resid = Vector{Float64}(undef, n))
    if a isa DiscreteActions
        length(res.X_ind) == n || (res.X_ind = Vector{Int}(undef, n))
        for i in 1:n
            res.V[i], res.X_ind[i] =
                _s_wise_max_discrete!(cdp, ker, _row(s_nodes, i), C, fec)
            res.X[i] = a.vals[res.X_ind[i]]
        end
    elseif a isa ContinuousActions && _action_dim(a) > 1
        # Respect the inner solver used by `solve`: a :brent solve must not
        # have its policy re-evaluated through the derivative-based path
        dfecs = res.inner_solver == :foc && !_forces_brent(ker) ?
            _eval_dfecs(res.interp) : nothing
        dfecs === nothing || foreach(dfec -> set_coefs!(dfec, C), dfecs)
        for i in 1:n
            res.V[i] = _s_wise_max_multi!(cdp, ker, _row(s_nodes, i), C,
                                          fec, dfecs, view(res.X, i, :),
                                          dfecs !== nothing)
        end
    else
        s_wise_max!(_colloc(res), s_nodes, C, res.V, res.X, fec)
    end
    for i in 1:n
        res.resid[i] = res.V[i] - funeval_point!(fec, C, _row(s_nodes, i))
    end
    return res
end

evaluate!(res::CDPSolveResult) =
    evaluate!(res, FunEvalCache(res.interp.basis))

function set_eval_nodes!(
        res::CDPSolveResult{Algo,1}, s_nodes_coord::NTuple{1,Vector{Float64}}
    ) where {Algo}
    s_nodes = s_nodes_coord[1]
    res.eval_nodes = s_nodes
    res.eval_nodes_coord = s_nodes_coord
    evaluate!(res)
end

function set_eval_nodes!(
        res::CDPSolveResult{Algo,N}, s_nodes_coord::NTuple{N,Vector{Float64}}
    ) where {Algo,N}
    s_nodes = gridmake(s_nodes_coord...)
    res.eval_nodes = s_nodes
    res.eval_nodes_coord = s_nodes_coord
    evaluate!(res)
end

function set_eval_nodes!(
        res::CDPSolveResult{Algo,N}, s_nodes_coord::NTuple{N,AbstractVector}
    ) where {Algo,N}
    T = Float64
    s_nodes_coord_vecs =
        ntuple(i -> collect(T, s_nodes_coord[i]), N)::NTuple{N,Vector{T}}
    set_eval_nodes!(res, s_nodes_coord_vecs)
end

function set_eval_nodes!(
        res::CDPSolveResult{Algo,N}, s_nodes_coord::Vararg{AbstractVector,N}
    ) where {Algo,N}
    set_eval_nodes!(res, s_nodes_coord)
end

@doc """
    set_eval_nodes!(res, s_nodes_coord)

Set the evaluation nodes and recompute the value/policy functions.

# Arguments

- `res::CDPSolveResult`: Solution object to update in place.
- `s_nodes_coord::NTuple{N,AbstractVector}`: Coordinate vectors of the new
  evaluation nodes.
""" set_eval_nodes!

"""
    (res::CDPSolveResult)(s_nodes)

Evaluate the solved model at user-supplied state nodes.

Returns `(V, X, resid)`, where `V` is the value function, `X` is the greedy
policy, and `resid` is the Bellman residual at `s_nodes`.
"""
function (res::CDPSolveResult)(s_nodes::AbstractArray{Float64})
    cdp, C = res.cdp, res.C
    a = cdp.actions
    ker = _build_kernel(_colloc(res))
    fec = FunEvalCache(res.interp.basis)
    n = size(s_nodes, 1)
    if a isa DiscreteActions
        V = Vector{Float64}(undef, n)
        X = Vector{_policy_eltype(a)}(undef, n)
        for i in 1:n
            V[i], k = _s_wise_max_discrete!(cdp, ker, _row(s_nodes, i), C,
                                            fec)
            X[i] = a.vals[k]
        end
    elseif a isa ContinuousActions && _action_dim(a) > 1
        V = Vector{Float64}(undef, n)
        X = _policy_buffer(a, n)
        dfecs = res.inner_solver == :foc && !_forces_brent(ker) ?
            _eval_dfecs(res.interp) : nothing
        dfecs === nothing || foreach(dfec -> set_coefs!(dfec, C), dfecs)
        for i in 1:n
            V[i] = _s_wise_max_multi!(cdp, ker, _row(s_nodes, i), C, fec,
                                      dfecs, view(X, i, :),
                                      dfecs !== nothing)
        end
    else
        V, X = s_wise_max(_colloc(res), s_nodes, C, fec)
    end
    resid = Vector{Float64}(undef, n)
    for i in 1:n
        resid[i] = V[i] - funeval_point!(fec, C, _row(s_nodes, i))
    end
    return V, X, resid
end


"""
    CDPWorkspace{TF,TD}

Preallocated buffers used by the solution algorithms for a dynamic program
bound to an interpolation scheme. Construct with
`CDPWorkspace(cp; inner_solver=:foc)`.

Not thread-safe: use one workspace per thread.

# Fields

- `fec::TF<:FunEvalCache`: Workspace for point evaluation of the value
  function.
- `dfecs::TD`: Tuple of `DerivFunEvalCache`s for the gradient of the value
  function (one per state dimension), or `nothing` if the
  first-order-condition solver does not apply (basis not continuously
  differentiable, or action-dependent shock weights; see below).
- `Tv::Vector{Float64}`: Buffer for updated values at the interpolation
  nodes.
- `X::TX<:VecOrMat{Float64}`: Buffer for updated actions at the
  interpolation nodes for a continuous action space, an `n x M` matrix for
  `M`-dimensional actions (initialized to `NaN`; also serves as the warm
  start for the inner maximization in the next sweep).
- `X_ind::Vector{Int}`: Buffer for updated action indices at the
  interpolation nodes for a discrete action space; empty otherwise.
- `inner_solver::Symbol`: `:foc` to solve the inner maximization via its
  first-order condition (with Brent as automatic fallback), or `:brent` to
  always use derivative-free Brent maximization. Has no effect for a
  discrete action space (solved by enumeration), but is still validated.
"""
struct CDPWorkspace{TF<:FunEvalCache,TD,TX<:VecOrMat{Float64}}
    fec::TF
    dfecs::TD
    Tv::Vector{Float64}
    X::TX
    X_ind::Vector{Int}
    inner_solver::Symbol
end

# The FOC solver requires the fitted value function to be continuously
# differentiable: any Chebyshev basis, or splines of degree >= 2. For
# piecewise linear bases the derivative is a step function, on which
# root-finding is meaningless.
_foc_suitable(p::ChebParams) = length(p) >= 2
_foc_suitable(p::SplineParams) = p.k >= 2
_foc_suitable(p::BasisParams) = false

"""
    CDPWorkspace(cp::_CollocationProblem; inner_solver=:foc)

Construct a `CDPWorkspace` for the bound problem `cp`. See [`solve`](@ref)
for the meaning of `inner_solver`.
"""
function CDPWorkspace(cp::_CollocationProblem{N};
                      inner_solver::Symbol=:foc) where N
    inner_solver in (:foc, :brent) ||
        throw(ArgumentError("inner_solver must be :foc or :brent"))
    _validate_weights(cp)
    cdp, interp = cp.cdp, cp.interp
    basis = interp.basis
    n = interp.length
    discrete = cdp.actions isa DiscreteActions
    dfecs = if !discrete && inner_solver == :foc &&
               all(d -> _foc_suitable(basis.params[d]), 1:N) &&
               !_forces_brent(_build_kernel(cp))
        ntuple(d -> DerivFunEvalCache(
                   basis, ntuple(i -> Int(i == d), Val(N))), Val(N))
    else
        nothing
    end
    return CDPWorkspace(
        FunEvalCache(basis),
        dfecs,
        Vector{Float64}(undef, n),
        discrete ? Float64[] : _policy_buffer(cdp.actions, n),
        discrete ? Vector{Int}(undef, n) : Int[],
        inner_solver
    )
end


#= Methods =#

# Non-copying access to the i-th point of a set of points stored as a
# Vector (one point = one scalar) or a Matrix (one point = one row).
_row(A::AbstractVector, i::Int) = A[i]
_row(A::AbstractMatrix, i::Int) = view(A, i, :)

"""
    s_wise_max!(cp, ss, C, Tv[, fec])

Find optimal value for each grid point. These helpers apply to
one-dimensional continuous action spaces; discrete and multi-dimensional
action spaces are handled internally by the operators.

# Arguments

- `cp::_CollocationProblem`: The dynamic program bound to its interpolation
  scheme.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `Tv::Vector{Float64}`: A buffer array to hold the updated value function.
- `fec::FunEvalCache`: Workspace for point evaluation of the value function.
  Constructed internally if not given.

# Returns

- `Tv::Vector{Float64}`: Updated value function vector.
"""
function s_wise_max!(cp::_CollocationProblem, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64},
                     fec::FunEvalCache)
    cdp = cp.cdp
    ker = _build_kernel(cp)
    n = size(ss, 1)
    for i in 1:n
        Tv[i], _ = _s_wise_max!(cdp, ker, _row(ss, i), C, fec)
    end
    return Tv
end

s_wise_max!(cp::_CollocationProblem, ss::AbstractArray{Float64},
            C::Vector{Float64}, Tv::Vector{Float64}) =
    s_wise_max!(cp, ss, C, Tv, FunEvalCache(cp.interp.basis))

"""
    s_wise_max!(cp, ss, C, Tv, X[, fec])

Find optimal value and action for each grid point.

# Arguments

- `cp::_CollocationProblem`: The dynamic program bound to its interpolation
  scheme.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `Tv::Vector{Float64}`: A buffer array to hold the updated value function.
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.
- `fec::FunEvalCache`: Workspace for point evaluation of the value function.
  Constructed internally if not given.

# Returns

- `Tv::Vector{Float64}`: Updated value function vector.
- `X::Vector{Float64}`: Updated policy function vector.
"""
function s_wise_max!(cp::_CollocationProblem, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64},
                     X::Vector{Float64}, fec::FunEvalCache)
    cdp = cp.cdp
    ker = _build_kernel(cp)
    n = size(ss, 1)
    for i in 1:n
        Tv[i], X[i] = _s_wise_max!(cdp, ker, _row(ss, i), C, fec)
    end
    return Tv, X
end

s_wise_max!(cp::_CollocationProblem, ss::AbstractArray{Float64},
            C::Vector{Float64}, Tv::Vector{Float64}, X::Vector{Float64}) =
    s_wise_max!(cp, ss, C, Tv, X, FunEvalCache(cp.interp.basis))

"""
    s_wise_max(cp, ss, C[, fec])

Find optimal value and action for each grid point.

# Arguments

- `cp::_CollocationProblem`: The dynamic program bound to its interpolation
  scheme.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `fec::FunEvalCache`: Workspace for point evaluation of the value function.
  Constructed internally if not given.

# Returns

- `Tv::Vector{Float64}`: Value function vector.
- `X::Vector{Float64}`: Policy function vector.
"""
function s_wise_max(cp::_CollocationProblem, ss::AbstractArray{Float64},
                    C::Vector{Float64}, fec::FunEvalCache)
    n = size(ss, 1)
    Tv, X = Array{Float64}(undef, n), Array{Float64}(undef, n)
    s_wise_max!(cp, ss, C, Tv, X, fec)
end

s_wise_max(cp::_CollocationProblem, ss::AbstractArray{Float64},
           C::Vector{Float64}) =
    s_wise_max(cp, ss, C, FunEvalCache(cp.interp.basis))


"""
    bellman_operator!(cp, C, Tv)
    bellman_operator!(cp, C, ws)

Apply the Bellman operator and update the basis coefficients. Values are
stored in `Tv` (or `ws.Tv`).

# Arguments

- `cp::_CollocationProblem`: The dynamic program bound to its interpolation
  scheme.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `Tv::Vector{Float64}`: Vector to store values.
- `ws::CDPWorkspace`: Workspace for the solution algorithms.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function bellman_operator!(cp::_CollocationProblem, C::Vector{Float64},
                           ws::CDPWorkspace)
    cdp, interp = cp.cdp, cp.interp
    if cdp.actions isa DiscreteActions
        _s_wise_max_discrete_sweep!(cp, C, ws.Tv, ws.X_ind, ws.fec)
    elseif cdp.actions isa ContinuousActions && _action_dim(cdp.actions) > 1
        _s_wise_max_multi_sweep!(cp, C, ws.Tv, ws.X, ws.fec, ws.dfecs,
                                 ws.inner_solver == :foc)
    elseif _use_foc(ws)
        _s_wise_max_foc_sweep!(cp, C, ws.Tv, ws.X, ws.fec, ws.dfecs)
    else
        s_wise_max!(cp, interp.S, C, ws.Tv, ws.X, ws.fec)
    end
    ldiv!(C, interp.Phi_lu, ws.Tv)
    return C
end

function bellman_operator!(cp::_CollocationProblem, C::Vector{Float64},
                           Tv::Vector{Float64})
    cdp, interp = cp.cdp, cp.interp
    if cdp.actions isa DiscreteActions
        X_ind = Vector{Int}(undef, length(Tv))
        _s_wise_max_discrete_sweep!(cp, C, Tv, X_ind,
                                    FunEvalCache(interp.basis))
    elseif cdp.actions isa ContinuousActions && _action_dim(cdp.actions) > 1
        ws = CDPWorkspace(cp)
        _s_wise_max_multi_sweep!(cp, C, Tv, ws.X, ws.fec, ws.dfecs,
                                 ws.inner_solver == :foc)
    else
        s_wise_max!(cp, interp.S, C, Tv)
    end
    ldiv!(C, interp.Phi_lu, Tv)
    return C
end


"""
    compute_greedy!(cp, C, X)
    compute_greedy!(cp, ss, C, X[, fec])

Compute the greedy policy for the given basis coefficients.

# Arguments

- `cp::_CollocationProblem`: The dynamic program bound to its interpolation
  scheme.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.
- `fec::FunEvalCache`: Workspace for point evaluation of the value function.
  Constructed internally if not given.

# Returns

- `X::Vector{Float64}`: Updated policy function vector.
"""
function compute_greedy!(cp::_CollocationProblem, ss::AbstractArray{Float64},
                         C::Vector{Float64}, X::Vector{Float64},
                         fec::FunEvalCache)
    cdp = cp.cdp
    ker = _build_kernel(cp)
    n = size(ss, 1)
    for i in 1:n
        _, X[i] = _s_wise_max!(cdp, ker, _row(ss, i), C, fec)
    end
    return X
end

compute_greedy!(cp::_CollocationProblem, ss::AbstractArray{Float64},
                C::Vector{Float64}, X::Vector{Float64}) =
    compute_greedy!(cp, ss, C, X, FunEvalCache(cp.interp.basis))

compute_greedy!(cp::_CollocationProblem, C::Vector{Float64},
                X::Vector{Float64}) =
    compute_greedy!(cp, cp.interp.S, C, X)

"""
    evaluate_policy!(cp, X, C[, fec])

Compute the value function for a given policy and update the basis
coefficients: solve `(Phi - beta * E[Phi(g(S, X, e))]) C = f(S, X)`, where
the expected-basis matrix is assembled row by row with the non-allocating
point-evaluation kernels. When the collocation matrix `Phi` is sparse
(spline and piecewise linear bases), the system matrix is assembled and
factorized in sparse form.

# Arguments

- `cp::_CollocationProblem`: The dynamic program bound to its interpolation
  scheme.
- `X::AbstractVecOrMat`: Policy function (action values); an `n x M`
  matrix for `M`-dimensional continuous actions.
- `C::Vector{Float64}`: A buffer array to hold the basis coefficients.
- `fec::FunEvalCache`: Workspace whose per-dimension caches are used for
  basis evaluation at the next states. Constructed internally if not given.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function evaluate_policy!(cp::_CollocationProblem, X::AbstractVecOrMat,
                          C::Vector{Float64}, fec::FunEvalCache)
    cdp, interp = cp.cdp, cp.interp
    A_lu = _policy_system_lu(interp.Phi, cp, X, fec)
    ss = interp.S
    for i in 1:size(ss, 1)
        C[i] = cdp.f(_row(ss, i), _row(X, i))
    end
    ldiv!(A_lu, C)
    return C
end

evaluate_policy!(cp::_CollocationProblem, X::AbstractVecOrMat,
                 C::Vector{Float64}) =
    evaluate_policy!(cp, X, C, FunEvalCache(cp.interp.basis))

"""
    policy_iteration_operator!(cp, C, X)
    policy_iteration_operator!(cp, C, ws)

Perform one step of policy function iteration and update the basis coefficients.

# Arguments

- `cp::_CollocationProblem`: The dynamic program bound to its interpolation
  scheme.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.
- `ws::CDPWorkspace`: Workspace for the solution algorithms.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function policy_iteration_operator!(cp::_CollocationProblem,
                                    C::Vector{Float64}, ws::CDPWorkspace)
    cdp = cp.cdp
    if cdp.actions isa DiscreteActions
        _s_wise_max_discrete_sweep!(cp, C, ws.Tv, ws.X_ind, ws.fec)
        evaluate_policy!(cp, view(cdp.actions.vals, ws.X_ind), C, ws.fec)
    elseif cdp.actions isa ContinuousActions && _action_dim(cdp.actions) > 1
        _s_wise_max_multi_sweep!(cp, C, ws.Tv, ws.X, ws.fec, ws.dfecs,
                                 ws.inner_solver == :foc)
        evaluate_policy!(cp, ws.X, C, ws.fec)
    elseif _use_foc(ws)
        _s_wise_max_foc_sweep!(cp, C, ws.Tv, ws.X, ws.fec, ws.dfecs)
        evaluate_policy!(cp, ws.X, C, ws.fec)
    else
        compute_greedy!(cp, cp.interp.S, C, ws.X, ws.fec)
        evaluate_policy!(cp, ws.X, C, ws.fec)
    end
    return C
end

function policy_iteration_operator!(cp::_CollocationProblem,
                                    C::Vector{Float64}, X::Vector{Float64})
    compute_greedy!(cp, C, X)
    evaluate_policy!(cp, X, C)
    return C
end


# Sup distance between two arrays, without allocating a temporary.
# NaN entries propagate to the result (as with `maximum(abs, x - y)`), so
# that a diverged iteration is never mistaken for a converged one.
function _max_abs_diff(x, y)
    err = abs(zero(promote_type(eltype(x), eltype(y))))
    @inbounds for i in eachindex(x, y)
        err = max(err, abs(x[i] - y[i]))
    end
    return err
end

"""
    operator_iteration!(T, C, tol, max_iter; verbose=2, print_skip=50)

Iterate an operator on the basis coefficients until convergence.

# Arguments

- `T::Function`: Operator that updates basis coefficients (one step of VFI or
  PFI).
- `C::Vector{Float64}`: Initial basis coefficient vector.
- `tol::Float64`: Convergence tolerance.
- `max_iter::Integer`: Maximum number of iterations.
- `verbose::Integer`: Level of feedback (0 for no output, 1 for warnings only,
  2 for warning and convergence messages during iteration).
- `print_skip::Integer`: If `verbose == 2`, how many iterations between print
  messages.

# Returns

- `converged::Bool`: Whether the iteration converged.
- `i::Int`: Number of iterations performed.
"""
function operator_iteration!(T::Function, C::TC, tol::Float64,
                             max_iter::Integer;
                             verbose::Integer=2,
                             print_skip::Integer=50) where TC
    converged = false
    i = 0

    if max_iter <= 0
        if verbose >= 1
            @warn("No computation performed with max_iter=$max_iter")
        end
        return converged, i
    end

    err = tol + 1
    C_old = similar(C)
    while true
        copyto!(C_old, C)
        C = T(C)::TC
        err = _max_abs_diff(C, C_old)
        i += 1
        (err <= tol) && (converged = true)

        (converged || i >= max_iter) && break

        if (verbose == 2) && (i % print_skip == 0)
            println("Compute iterate $i with error $err")
        end
    end

    if verbose == 2
        println("Compute iterate $i with error $err")
    end

    if verbose >= 1
        if !converged
            @warn("max_iter attained")
        elseif verbose == 2
            println("Converged in $i steps")
        end
    end

    return converged, i
end


#= Solver types =#

"""
    CollocationSolver(basis; algorithm=PFI, inner_solver=:foc,
                      tol=sqrt(eps()), max_iter=500)
    CollocationSolver(; basis, algorithm=PFI, inner_solver=:foc,
                      tol=sqrt(eps()), max_iter=500)

Bellman equation collocation solver configuration for `solve`: the
interpolation basis together with the algorithm parameters. The problem
itself (a [`ContinuousDP`](@ref)) holds only the model primitives.

# Arguments

- `basis::Basis`: Object that contains the interpolation basis information;
  its domain is the approximation domain of the value function.
- `algorithm::Type{<:DPAlgorithm}`: `PFI` for policy function iteration or
  `VFI` for value function iteration (for linear-quadratic approximation
  see [`LQASolver`](@ref)).
- `inner_solver::Symbol`: How to solve the inner maximization over actions.
  `:foc` (default) solves the first-order condition by safeguarded
  root-finding, warm-started across iterations, using the exact gradient of
  the fitted value function and finite differences of `f` and `g`. It is
  intended for smooth, effectively concave inner problems where the
  first-order condition identifies the maximizing action: it falls back to
  Brent maximization state-by-state when derivative evaluation is
  unavailable or non-finite (and is used only for continuously
  differentiable bases: any Chebyshev, or splines of degree >= 2), but it
  does not attempt to detect nonconcavity or multiple stationary points.
  Use `inner_solver=:brent` for the derivative-free path. The choice has
  no effect for discrete action spaces (solved by exact enumeration), but
  the value is still validated. With action-dependent shock weights
  `weights(s, x)` (see [`ContinuousDP`](@ref)) the first-order condition
  acquires a term the FOC solver does not compute, so `:foc` automatically
  behaves as `:brent`.
- `tol::Real`: Convergence tolerance.
- `max_iter::Integer`: Maximum number of iterations.
"""
struct CollocationSolver{Algo<:DPAlgorithm,TB<:Basis}
    basis::TB
    inner_solver::Symbol
    tol::Float64
    max_iter::Int

    function CollocationSolver{Algo,TB}(basis, inner_solver, tol,
                                        max_iter) where {Algo<:DPAlgorithm,
                                                         TB<:Basis}
        (Algo === PFI || Algo === VFI) || throw(ArgumentError(
            "algorithm must be PFI or VFI; use LQASolver for " *
            "linear-quadratic approximation"))
        inner_solver in (:foc, :brent) ||
            throw(ArgumentError("inner_solver must be :foc or :brent"))
        # Validate the converted values that are stored: Float64 conversion
        # of an extreme Real (e.g. BigFloat) can underflow to 0.0 or
        # overflow to Inf even when the raw value is positive and finite
        tol_f = Float64(tol)
        (isfinite(tol_f) && tol_f > 0) ||
            throw(ArgumentError("tol must be a positive finite number"))
        max_iter_i = Int(max_iter)
        # max_iter == 0 is meaningful: fit v_init and iterate zero times
        max_iter_i >= 0 ||
            throw(ArgumentError("max_iter must be nonnegative"))
        return new{Algo,TB}(basis, inner_solver, tol_f, max_iter_i)
    end
end

CollocationSolver(basis::Basis;
                  algorithm::Type{<:DPAlgorithm}=PFI,
                  inner_solver::Symbol=:foc,
                  tol::Real=sqrt(eps()), max_iter::Integer=500) =
    CollocationSolver{algorithm,typeof(basis)}(basis, inner_solver, tol,
                                               max_iter)

CollocationSolver(; basis, kwargs...) = CollocationSolver(basis; kwargs...)

"""
    LQASolver(basis; point)
    LQASolver(; basis, point)

Linear-quadratic approximation solver configuration for `solve`: the model
is approximated around the reference point `point = (s, x, e)` and the
resulting LQ problem is solved exactly; the value function of the LQ
solution is then represented in the interpolation `basis` (so the result
has the same interface as a collocation solution).

# Arguments

- `basis::Basis`: Object that contains the interpolation basis information.
- `point::Tuple`: The reference point `(s, x, e)` (typically a steady
  state) around which the LQ approximation is constructed.
"""
struct LQASolver{TB<:Basis,
                 TP<:Tuple{ScalarOrArray,ScalarOrArray,ScalarOrArray}}
    basis::TB
    point::TP
end

LQASolver(basis::Basis; point) = LQASolver(basis, point)
LQASolver(; basis, point) = LQASolver(basis, point)


#= Solve methods =#

"""
    solve(cdp, solver)

Solve the continuous-state dynamic program `cdp` with the given solver
configuration ([`CollocationSolver`](@ref) or [`LQASolver`](@ref)).

# Arguments

- `cdp::ContinuousDP`: The dynamic program to solve.
- `solver`: Solver configuration.
- `v_init::Vector{Float64}`: Optional keyword; initial value function values
  at the interpolation nodes of `solver.basis`.
- `verbose::Integer`: Optional keyword; level of feedback (0 for no output,
  1 for warnings only, 2 for warnings and convergence messages during
  iteration).
- `print_skip::Integer`: Optional keyword; if `verbose == 2`, how many
  iterations between print messages.

# Returns

- `res::CDPSolveResult`: Solution object of the dynamic program.
"""
function solve(cdp::ContinuousDP, solver::CollocationSolver{Algo};
               v_init=nothing, verbose::Integer=2,
               print_skip::Integer=50) where {Algo<:DPAlgorithm}
    interp = Interp(solver.basis)
    return _solve_core(_CollocationProblem(cdp, interp), Algo,
                       _check_v_init(v_init, interp),
                       solver.tol, solver.max_iter, solver.inner_solver,
                       verbose, print_skip)
end

function solve(cdp::ContinuousDP, solver::LQASolver;
               v_init=nothing, verbose::Integer=2, print_skip::Integer=50)
    interp = Interp(solver.basis)
    return _solve_core(_CollocationProblem(cdp, interp), LQA,
                       _check_v_init(v_init, interp),
                       sqrt(eps()), 500, :brent,
                       verbose, print_skip; point=solver.point)
end

_check_v_init(::Nothing, interp::Interp) = zeros(interp.length)
function _check_v_init(v_init::AbstractVector, interp::Interp)
    length(v_init) == interp.length || throw(ArgumentError(
        "length(v_init) = $(length(v_init)) does not match the number of " *
        "interpolation nodes of the solver's basis ($(interp.length))"))
    return convert(Vector{Float64}, v_init)
end

# Shared solve pipeline over a problem with a bound interpolation scheme
function _solve_core(cp::_CollocationProblem, ::Type{Algo},
                     v_init::Vector{Float64}, tol::Float64, max_iter::Int,
                     inner_solver::Symbol, verbose::Integer,
                     print_skip::Integer;
                     kwargs...) where {Algo<:DPAlgorithm}
    res = CDPSolveResult{Algo}(cp.cdp, cp.interp, tol, max_iter,
                               inner_solver)
    # LQA has no inner maximization: skip the FOC derivative caches
    ws = CDPWorkspace(cp; inner_solver=(Algo === LQA ? :brent : inner_solver))
    ldiv!(res.C, cp.interp.Phi_lu, v_init)
    _solve!(cp, res, ws, verbose, print_skip; kwargs...)
    evaluate!(res, ws.fec)
    return res
end

# TODO: delete this error stub in a later release (kept for migration
# guidance from the v0.2 API, removed in v0.3)
solve(cdp::ContinuousDP, ::Type{<:DPAlgorithm}=PFI; kwargs...) =
    throw(ArgumentError(
        "`solve(cdp, PFI/VFI/LQA; ...)` has been removed: pass a " *
        "solver object, e.g. `solve(cdp, CollocationSolver(basis))` or " *
        "`solve(cdp, LQASolver(basis; point=point))`"))


# Policy iteration
@doc """
    PFI

Policy function iteration algorithm for `solve`.
"""
PFI

"""
    _solve!(cp, res, ws, verbose, print_skip)

Implement policy iteration. See `solve` for further details.
"""
function _solve!(cp::_CollocationProblem, res::CDPSolveResult{PFI},
                 ws::CDPWorkspace, verbose, print_skip)
    operator!(C) = policy_iteration_operator!(cp, C, ws)
    res.converged, res.num_iter =
        operator_iteration!(operator!, res.C, res.tol, res.max_iter,
                            verbose=verbose, print_skip=print_skip)
    return res
end


# Value iteration
@doc """
    VFI

Value function iteration algorithm for `solve`.
"""
VFI

"""
    _solve!(cp, res, ws, verbose, print_skip)

Implement value iteration. See `solve` for further details.
"""
function _solve!(cp::_CollocationProblem, res::CDPSolveResult{VFI},
                 ws::CDPWorkspace, verbose, print_skip)
    operator!(C) = bellman_operator!(cp, C, ws)
    res.converged, res.num_iter =
        operator_iteration!(operator!, res.C, res.tol, res.max_iter,
                            verbose=verbose, print_skip=print_skip)
    return res
end

# Linear Quadratic Approximation
"""
    LQA

Linear-quadratic approximation algorithm for `solve`.

Used via `solve(cdp, LQASolver(basis; point=(s, x, e)))` to approximate the
model around a reference point and solve the resulting LQ problem.
"""
struct LQA <: DPAlgorithm end

"""
    _solve!(cp, res, ws, verbose, print_skip; point)

Implement linear quadratic approximation. See `solve` for further details.
"""
function _solve!(cp::_CollocationProblem,
                 res::CDPSolveResult{LQA},
                 ws::CDPWorkspace, verbose, print_skip;
                 point::Tuple{ScalarOrArray, ScalarOrArray, ScalarOrArray})
    cdp, interp = cp.cdp, cp.interp
    cdp.actions isa ContinuousActions || throw(ArgumentError(
        "LQA requires a continuous action space"))

    # Unpack point
    s_star, x_star, e_star = point

    jacobian = FiniteDiff.finite_difference_jacobian
    gradient = FiniteDiff.finite_difference_gradient
    hessian = FiniteDiff.finite_difference_hessian

    # Compute zero-th order terms
    f_star = cdp.f(s_star, x_star);
    g_star = cdp.g(s_star, x_star, e_star);
    z_star = [s_star..., x_star...]

    # Compute derivatives -- need to handle scalar and vector cases separately
    s_is_nb = isa(s_star, Number)
    x_is_nb = isa(x_star, Number)

    n = length(s_star)
    m = length(x_star)
    s_range = s_is_nb ? 1 : 1:n
    x_range = x_is_nb ? n+1 : n+1:n+m

    f_vec = z -> cdp.f(z[s_range], z[x_range])
    g_vec = z -> cdp.g(z[s_range], z[x_range], e_star)

    Df_star = gradient(f_vec, z_star)
    DDf_star = hessian(f_vec, z_star)
    Dg_star = s_is_nb ? gradient(g_vec, z_star)' : jacobian(g_vec, z_star)

    # Construct LQ approximation instance
    lq = approx_lq(s_star, x_star, f_star, Df_star, DDf_star, g_star, Dg_star,
                   cdp.discount)

    # Solve LQ problem
    P, F, d = stationary_values(lq)

    # Compute value function
    v(s) = -([1, s...]' * P * [1, s...] + d)
    v_vals = [v(interp.S[i, :]) for i in 1:length(interp.basis)]

    # Back out basis coefficients
    ldiv!(res.C, interp.Phi_lu, v_vals)

    res.converged = true
end

#= Point evaluation of the solution: value and policy functors =#

"""
    ValueFunction(res::CDPSolveResult)

Callable object evaluating the fitted value function at a single state
point: `vf = ValueFunction(res); vf(s)`. Evaluation is non-allocating; the
basis coefficients are shared with `res` (not copied).

Not thread-safe: use one instance per thread (as with [`CDPWorkspace`](@ref)).
"""
struct ValueFunction{TF<:FunEvalCache}
    C::Vector{Float64}
    fec::TF
end

ValueFunction(res::CDPSolveResult) =
    ValueFunction(res.C, FunEvalCache(res.interp.basis))

(vf::ValueFunction)(s) = funeval_point!(vf.fec, vf.C, s)

"""
    PolicyFunction(res::CDPSolveResult)

Callable object evaluating the policy function at a single state point:
`pf = PolicyFunction(res); pf(s)`. For a discrete action space, the greedy
action is recomputed exactly at `s` (a discrete policy is never
interpolated); for a continuous action space, the policy values `res.X` are
interpolated piecewise linearly over the evaluation nodes and clamped into
`[x_lb(s), x_ub(s)]`. Returns an action value for scalar actions and a
length-`M` tuple for `M`-dimensional continuous actions.

The policy data are shared with `res` at construction time: for continuous
actions, construct after the final `set_eval_nodes!` call. The evaluator
machinery is allocation-free; total per-call allocations depend on the
user-supplied functions invoked during evaluation (the action-bound
functions for continuous actions; the reward and transition functions for
discrete ones). Not thread-safe: use one instance per thread.
"""
abstract type PolicyFunction end

# Discrete actions: exact greedy recomputation at the evaluation point
struct _GreedyPolicyFunction{TCDP<:ContinuousDP,TK,TF<:FunEvalCache,
                             TV<:AbstractVector} <: PolicyFunction
    cdp::TCDP
    ker::TK
    C::Vector{Float64}
    fec::TF
    vals::TV
end

(pf::_GreedyPolicyFunction)(s) =
    pf.vals[_s_wise_max_discrete!(pf.cdp, pf.ker, s, pf.C, pf.fec)[2]]

# Continuous actions: piecewise-linear interpolation of the nodal policy
# values, evaluated with the non-allocating point kernel. For a piecewise
# linear basis with breakpoints at the evaluation nodes, the interpolant's
# coefficients coincide with the nodal values, so `res.X` is used as the
# coefficient array directly.
struct _InterpPolicyFunction{M,TA<:ContinuousActions,TF<:FunEvalCache,
                             TX<:AbstractVecOrMat} <: PolicyFunction
    actions::TA
    X::TX
    fec::TF
end

function (pf::_InterpPolicyFunction{1})(s)
    x = funeval_point!(pf.fec, pf.X, s)
    return clamp(x, pf.actions.x_lb(s), pf.actions.x_ub(s))
end

function (pf::_InterpPolicyFunction{M})(s) where {M}
    lb, ub = pf.actions.x_lb(s), pf.actions.x_ub(s)
    return ntuple(
        d -> clamp(funeval_point!(pf.fec, view(pf.X, :, d), s), lb[d], ub[d]),
        Val(M))
end

function PolicyFunction(res::CDPSolveResult{Algo,N}) where {Algo,N}
    a = res.cdp.actions
    if a isa DiscreteActions
        return _GreedyPolicyFunction(res.cdp, _build_kernel(_colloc(res)),
                                     res.C,
                                     FunEvalCache(res.interp.basis),
                                     a.vals)
    end
    basis = Basis(map(LinParams, res.eval_nodes_coord, ntuple(i -> 0, N)))
    fec = FunEvalCache(basis)
    M = _action_dim(a)
    return _InterpPolicyFunction{M,typeof(a),typeof(fec),typeof(res.X)}(
        a, res.X, fec)
end


#= Simulate methods =#

"""
    simulate!([rng=GLOBAL_RNG], s_path, res, s_init)

Generate a sample path of state variable(s) from a solved model.

# Arguments

- `rng::AbstractRNG`: Random number generator.
- `s_path::VecOrMat`: Array to store the generated sample path.
- `res::CDPSolveResult`: Solution object of the dynamic program.
- `s_init`: Initial value of state variable(s).

# Returns

- `s_path::VecOrMat`: Generated sample path of state variable(s).
"""
function simulate!(rng::AbstractRNG, s_path::VecOrMat,
                   res::CDPSolveResult{Algo,N},
                   s_init) where {Algo,N}
    ts_length = size(s_path)[end]

    # Policy lookup: exact greedy recomputation for a discrete action
    # space, piecewise-linear interpolation (with clamping into the action
    # bounds) for a continuous one; see PolicyFunction
    policy = PolicyFunction(res)

    s_ind_front = Base.front(axes(s_path))
    e_ind_tail = Base.tail(axes(res.cdp.shocks))
    view(s_path, (s_ind_front..., 1)... ) .= s_init

    if res.cdp.weights isa AbstractVector
        # Sampling needs a proper probability vector (the Bellman-side
        # sub-stochastic allowance has no path-wise meaning): the fixed
        # path validates once here; the callable path in the else branch
        # validates per visited (s, x) inside _draw_branch_index.
        # Draws are scaled by the validated total (sampling the
        # normalized distribution within the accepted tolerance), and the
        # index is clamped: r * total can round up to or above cdf[end],
        # which would otherwise index one past the last shock. (The clamp
        # also covers proper weights whose cumsum rounds slightly below
        # one — a latent out-of-bounds edge predating the kernel work.)
        total = _check_sampling_weights(res.cdp.weights)
        cdf = cumsum(res.cdp.weights)
        n_shocks = length(cdf)
        r = rand(rng, ts_length - 1)
        e_ind = Array{Int}(undef, ts_length - 1)
        for t in 1:ts_length - 1
            e_ind[t] = min(searchsortedlast(cdf, r[t] * total) + 1,
                           n_shocks)
        end
        for t in 1:ts_length - 1
            s = s_path[(s_ind_front..., t)...]
            x = policy(s)
            e = res.cdp.shocks[(e_ind[t], e_ind_tail...)...]
            view(s_path, (s_ind_front..., t + 1)... ) .= res.cdp.g(s, x, e)
        end
    else
        # State- or state-action-dependent weights: the branch
        # distribution depends on the visited path, so draw sequentially
        ker = _build_kernel(_colloc(res))
        for t in 1:ts_length - 1
            s = s_path[(s_ind_front..., t)...]
            x = policy(s)
            j = _draw_branch_index(rng, ker, s, x)
            e = res.cdp.shocks[(j, e_ind_tail...)...]
            view(s_path, (s_ind_front..., t + 1)... ) .= res.cdp.g(s, x, e)
        end
    end

    return s_path
end

simulate!(s_path::VecOrMat{Float64}, res::CDPSolveResult, s_init) =
    simulate!(Random.GLOBAL_RNG, s_path, res, s_init)

"""
    simulate([rng=GLOBAL_RNG], res, s_init, ts_length)

Generate a sample path of state variable(s) from a solved model.

# Arguments

- `rng::AbstractRNG`: Random number generator.
- `res::CDPSolveResult`: Solution object of the dynamic program.
- `s_init`: Initial value of state variable(s).
- `ts_length::Integer`: Length of simulation.

# Returns

- `s_path::VecOrMat`: Generated sample path of state variable(s).
"""
function simulate(rng::AbstractRNG, res::CDPSolveResult{Algo,1}, s_init::Real,
                  ts_length::Integer) where {Algo<:DPAlgorithm}
    s_path = Array{Float64}(undef, ts_length)
    simulate!(rng, s_path, res, s_init)
    return s_path
end

simulate(res::CDPSolveResult{Algo,1}, s_init::Real,
         ts_length::Integer) where {Algo<:DPAlgorithm} =
    simulate(Random.GLOBAL_RNG, res, s_init, ts_length)

function simulate(rng::AbstractRNG, res::CDPSolveResult, s_init::Vector,
                  ts_length::Integer)
    s_path = Array{Float64}(undef, length(s_init), ts_length)
    simulate!(rng, s_path, res, s_init)
    return s_path
end

simulate(res::CDPSolveResult, s_init::Vector, ts_length::Integer) =
    simulate(Random.GLOBAL_RNG, res, s_init, ts_length)
