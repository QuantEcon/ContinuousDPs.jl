"""
    _TransitionKernel

Abstract supertype of internal transition-kernel representations: the
distribution of the next state given `(s, x)`, as finitely many weighted
branches `(s'_k, w_k)`.

The general contract is:

- `_branch_sum(f, ker, s, x, args...)`: return the sum of
  `f(s', w, args...)` over the branches at `(s, x)`;
- `_foreach_branch(f, ker, s, x, args...)`: call `f(s', w, args...)` on
  each branch at `(s, x)` (the intended seam for side-effecting
  consumers such as a future generic policy-assembly path);
- `_forces_brent(ker)`: whether the first-order-condition inner solver
  must fall back to Brent. Defaults to `true`: the FOC paths additionally
  require the indexed access described under [`_QuadratureKernel`](@ref),
  which a general kernel need not provide.

These operations suffice for the generic Bellman expectation
(`_expected_value`), Brent maximization, and discrete-action
enumeration. They do NOT yet cover policy-system assembly or simulation:
both currently consume the structured tier only (`_policy_system_lu`
assembles rows with the indexed primitives directly, and `simulate!`
draws a branch index with the `_QuadratureKernel` helper
`_draw_branch_index` and maps it into the shock nodes). A general-kernel
sampling hook — a `_draw_next_state` returning the next state itself,
since a general kernel need not expose stable branch indices — is to be
introduced together with its first consumer.

In the traversals, `f` is a top-level function and `args` its explicit
payload. This avoids a capturing closure (which would materialize on the
heap here), but the generic traversal indirection is nevertheless not
allocation-free under the current implementation: measurements show
roughly 50-80 bytes per call. Allocation-sensitive structured consumers
therefore retain direct `_QuadratureKernel` specializations.

Opting into the FOC solvers by overriding `_forces_brent(ker) = false`
declares more than the traversal contract: such a kernel must
additionally provide the structured tier's indexed access
(`_branch_weights`, `_branch_state`) with a stable branch count and
stable branch identity under action perturbations, and branch
probabilities that do not depend on the action (the FOC solvers do not
compute probability derivatives).
"""
abstract type _TransitionKernel end

_forces_brent(::_TransitionKernel) = true

"""
    _QuadratureKernel{TG,TR,TW} <: _TransitionKernel

The structured transition kernel of a `ContinuousDP`: the transition
function `g` paired with fixed quadrature nodes and a weights carrier.
Branch `j` has next state `g(s, x, shocks[j])` and weight `w[j]`, where
`w` is the weight container at `(s, x)`: the fixed vector itself, or the
value of a user-supplied callable `weights(s)` / `weights(s, x)` (wrapped
in `_StateWeights` / `_StateActionWeights` by `_build_kernel`).

Besides the general `_TransitionKernel` contract, this kernel provides
indexed access — `_branch_weights(ker, s, x)` (the weight container,
fetched once per `(s, x)`) and `_branch_state(ker, s, x, j)` — on which
the FOC solvers rely: derivative-based consumers re-evaluate
`_branch_state` at perturbed actions holding the branch index fixed.
Hence `_forces_brent` is `false` for fixed or state-only weights, `true`
for action-dependent weights (whose derivative term the FOC solvers do
not compute). The sampling helper `_draw_branch_index(rng, ker, s, x)`
(used by `simulate!` for callable weights) also lives at this tier: it
returns an index into the fixed shock nodes.

Allocation contract: a callable weights function returning a `Tuple` or a
statically-sized vector (e.g. a `StaticArrays.SVector`) keeps the sweeps
allocation-free; returning a freshly allocated `Vector` is supported but
makes the path allocation-lean instead. Cost contract: the callable is
invoked on every objective evaluation of the inner maximization — several
times per state per sweep, not once per state — so it should be cheap;
anything expensive belongs in a table or interpolant precomputed outside
the callable.
"""
struct _QuadratureKernel{TG,TR<:AbstractVecOrMat,TW} <: _TransitionKernel
    g::TG
    shocks::TR
    weights::TW
end

# Callable-weights carriers, classified by arity at kernel construction.
# They carry the shock-node count so that every fetch can check the
# returned length: the one-time probe in _validate_weights may be
# legitimately skipped (infeasible probe point), and a wrong-length
# return would otherwise silently truncate the branch loop rather than
# error. For Tuple returns the length is a compile-time constant, so the
# check is a single integer compare on the hot path.
struct _StateWeights{F}         # weights(s)
    w::F
    n::Int
end
struct _StateActionWeights{F}   # weights(s, x)
    w::F
    n::Int
end

@noinline _weights_length_error(m, n) = throw(ArgumentError(
    "callable `weights` must return one weight per shock node ($n); " *
    "got $m"))

@inline function _checked_length(w, n::Int)
    length(w) == n || _weights_length_error(length(w), n)
    return w
end

_fetch_weights(w::AbstractVector, s, x) = w
_fetch_weights(sw::_StateWeights, s, x) = _checked_length(sw.w(s), sw.n)
_fetch_weights(sw::_StateActionWeights, s, x) =
    _checked_length(sw.w(s, x), sw.n)

# With action-dependent weights the first-order condition acquires a
# dH/dx term the FOC solvers do not compute: force the Brent fallback.
_forces_brent(ker::_QuadratureKernel) =
    ker.weights isa _StateActionWeights

# Branch weights at (s, x), as an indexable collection aligned with the
# branch indices. Fixed quadrature weights ignore (s, x).
_branch_weights(ker::_QuadratureKernel, s, x) =
    _fetch_weights(ker.weights, s, x)

# Next state on branch j at (s, x)
_branch_state(ker::_QuadratureKernel, s, x, j::Int) =
    ker.g(s, x, _row(ker.shocks, j))

# General-contract traversal, implemented by indexed access
@inline function _branch_sum(f::F, ker::_QuadratureKernel, s, x,
                             args::Vararg{Any,N}) where {F,N}
    w = _branch_weights(ker, s, x)
    acc = 0.0
    for j in eachindex(w)
        acc += f(_branch_state(ker, s, x, j), w[j], args...)
    end
    return acc
end

@inline function _foreach_branch(f::F, ker::_QuadratureKernel, s, x,
                                 args::Vararg{Any,N}) where {F,N}
    w = _branch_weights(ker, s, x)
    for j in eachindex(w)
        f(_branch_state(ker, s, x, j), w[j], args...)
    end
    return nothing
end

# E[V^(s')] under the kernel at (s, x), where V^ is the interpolant with
# coefficients C evaluated through fec. The general default goes through
# the traversal contract (allocation-lean: the traversal indirection is
# not free); the structured kernel overrides it with the direct indexed
# loop, which is measurably allocation-free.
_weighted_value(sp, w, fec, C) = w * funeval_point!(fec, C, sp)
_expected_value(ker::_TransitionKernel, fec::FunEvalCache, C, s, x) =
    _branch_sum(_weighted_value, ker, s, x, fec, C)

function _expected_value(ker::_QuadratureKernel, fec::FunEvalCache, C, s, x)
    w = _branch_weights(ker, s, x)
    cont = 0.0
    for j in eachindex(w)
        cont += w[j] * funeval_point!(fec, C, _branch_state(ker, s, x, j))
    end
    return cont
end

"""
    _build_kernel(cdp, s_probe, x_probe)
    _build_kernel(cp::_CollocationProblem)

Construct the `_QuadratureKernel` of `cdp`. For a fixed weight vector this
is free (a non-escaping bundle of references). For callable `weights` the
arity is detected by `hasmethod` (`weights(s, x)` takes precedence if both
apply); the user callable is never evaluated here — construction sits on
the sweep path, so return-shape validation happens once per solve instead,
in [`_validate_weights`](@ref) at workspace creation. Sweep-level entry
points construct the kernel once per call and pass it to the per-state
solvers.
"""
_build_kernel(cdp::ContinuousDP, s_probe, x_probe) =
    _build_kernel_w(cdp.weights, cdp, s_probe, x_probe)

_build_kernel_w(w::AbstractVector, cdp::ContinuousDP, s_probe, x_probe) =
    _QuadratureKernel(cdp.g, cdp.shocks, w)

function _build_kernel_w(w, cdp::ContinuousDP, s_probe, x_probe)
    n = size(cdp.shocks, 1)
    wrapped = if hasmethod(w, Tuple{typeof(s_probe),typeof(x_probe)})
        _StateActionWeights(w, n)
    elseif hasmethod(w, Tuple{typeof(s_probe)})
        _StateWeights(w, n)
    else
        throw(ArgumentError(
            "callable `weights` must accept a state `weights(s)` or a " *
            "state-action pair `weights(s, x)`"))
    end
    return _QuadratureKernel(cdp.g, cdp.shocks, wrapped)
end

"""
    _validate_weights(cp::_CollocationProblem)

One-time validation of callable `weights`, called at workspace creation
(once per solve) so that nothing here rides the sweep path: classify the
arity (throws for a callable of the wrong arity) and evaluate one probe
call at the first interpolation node and its probe action to validate
the returned container. The probe point is not guaranteed feasible, and
a model may legitimately error there (e.g. weights undefined for an
action with `f(s, x) == -Inf`, whose transition the solvers never
evaluate): if the probe call throws, this validation is skipped —
correctness then rests on the per-fetch length check built into the
weight carriers, so a malformed return still errors at its first actual
use instead of silently truncating the branch loop.
"""
function _validate_weights(cp::_CollocationProblem)
    cdp = cp.cdp
    cdp.weights isa AbstractVector && return nothing
    s_probe = _row(cp.interp.S, 1)
    x_probe = _probe_action(cdp.actions, s_probe)
    ker = _build_kernel(cdp, s_probe, x_probe)
    probe = try
        _raw_weights(ker.weights, s_probe, x_probe)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    # An actual `nothing` (or any non-collection) return reaches the
    # probe check and gets the informative error, unlike a thrown probe
    _check_weights_probe(probe, size(cdp.shocks, 1))
    return nothing
end

# The unchecked fetch, for the probe only: the probe check reports a
# wrong length with more context than the per-fetch guard
_raw_weights(sw::_StateWeights, s, x) = sw.w(s)
_raw_weights(sw::_StateActionWeights, s, x) = sw.w(s, x)

function _check_weights_probe(probe, n_shocks::Int)
    (probe isa Tuple || probe isa AbstractVector) || throw(ArgumentError(
        "callable `weights` must return an indexable collection of " *
        "weights (a Tuple, SVector, or Vector); got $(typeof(probe))"))
    all(wj -> wj isa Real, probe) || throw(ArgumentError(
        "callable `weights` must return real weights"))
    length(probe) == n_shocks || throw(ArgumentError(
        "callable `weights` must return one weight per shock node " *
        "($n_shocks); got $(length(probe))"))
    return nothing
end

_probe_action(a::DiscreteActions, s) = a.vals[1]
_probe_action(a::ContinuousActions{1}, s) = Float64(a.x_lb(s))
_probe_action(a::ContinuousActions{M}, s) where {M} =
    ntuple(d -> Float64(a.x_lb(s)[d]), Val(M))

function _build_kernel(cp::_CollocationProblem)
    cdp = cp.cdp
    # Fixed weights probe nothing: no user function is evaluated here
    cdp.weights isa AbstractVector &&
        return _QuadratureKernel(cdp.g, cdp.shocks, cdp.weights)
    s_probe = _row(cp.interp.S, 1)
    return _build_kernel(cdp, s_probe, _probe_action(cdp.actions, s_probe))
end

# Sampling requires a proper probability vector. The Bellman operators
# permit sub-stochastic weights (missing mass has zero continuation
# value, acting as extra discounting), but that interpretation has no
# path-wise counterpart: silently assigning the missing mass to a branch
# would simulate a different process than the one solved. Reject at the
# sampling boundary; return the total so the samplers can draw on
# [0, total) — within the accepted tolerance, sampling the normalized
# distribution rather than misassigning the residual mass.
function _check_sampling_weights(w)
    total = 0.0
    for wj in w
        (isfinite(wj) && wj >= 0) || throw(ArgumentError(
            "simulation requires finite nonnegative weights; got $wj"))
        total += wj
    end
    abs(total - 1) <= 1e-8 || throw(ArgumentError(
        "simulation requires weights summing to one; got $total. " *
        "Sub-stochastic weights are supported by the Bellman operators " *
        "(missing mass acts as extra discounting) but have no defined " *
        "sampling semantics"))
    return total
end

# Draw a branch index from the branch distribution at (s, x) by inverse
# CDF, accumulating the weights without materializing them. The draw is
# scaled by the validated total and compared with `r < acc`: the first
# branch j with r < w_1 + ... + w_j, the same convention as the
# fixed-weights path in simulate! (a zero draw never selects a leading
# zero-probability branch).
function _draw_branch_index(rng::AbstractRNG, ker::_QuadratureKernel, s, x)
    w = _branch_weights(ker, s, x)
    total = _check_sampling_weights(w)
    r = rand(rng) * total
    acc = 0.0
    last = 1
    for j in eachindex(w)
        wj = w[j]
        acc += wj
        wj > 0 && (last = j)
        r < acc && return j
    end
    return last  # r can round up to the total: last positive-weight branch
end
