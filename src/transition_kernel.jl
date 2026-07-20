"""
    _QuadratureKernel{TG,TR,TW}

Internal representation of the transition kernel of a `ContinuousDP`: the
distribution of the next state given `(s, x)`, as finitely many weighted
branches `(s'_j, w_j)`. This structured kernel pairs the transition
function `g` with fixed quadrature nodes and a weights carrier; branch `j`
has next state `g(s, x, shocks[j])` and weight `w[j]`, where `w` is the
weight container at `(s, x)`: the fixed vector itself, or the value of a
user-supplied callable `weights(s)` / `weights(s, x)` (wrapped in
`_StateWeights` / `_StateActionWeights` by `_build_kernel`).

Consumers iterate branches by index: fetch the weight container once per
`(s, x)` with `_branch_weights` and evaluate next states with
`_branch_state`; derivative-based consumers re-evaluate `_branch_state`
at perturbed actions holding the branch index fixed. The plain expected
value of an interpolant is provided by `_expected_value`.

Allocation contract: a callable weights function returning a `Tuple` or a
statically-sized vector (e.g. a `StaticArrays.SVector`) keeps the sweeps
allocation-free; returning a freshly allocated `Vector` is supported but
makes the path allocation-lean instead.
"""
struct _QuadratureKernel{TG,TR<:AbstractVecOrMat,TW}
    g::TG
    shocks::TR
    weights::TW
end

# Callable-weights carriers, classified by arity at kernel construction
struct _StateWeights{F}         # weights(s)
    w::F
end
struct _StateActionWeights{F}   # weights(s, x)
    w::F
end

_fetch_weights(w::AbstractVector, s, x) = w
_fetch_weights(sw::_StateWeights, s, x) = sw.w(s)
_fetch_weights(sw::_StateActionWeights, s, x) = sw.w(s, x)

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

# E[V^(s')] under the kernel at (s, x), where V^ is the interpolant with
# coefficients C evaluated through fec
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
arity is detected (`weights(s, x)` takes precedence if both apply) and the
callable is evaluated once at the probe point to validate its return: an
indexable collection of `Real`s with one weight per shock node. The
`_CollocationProblem` method probes at the first interpolation node and
the lower-bound action there. Sweep-level entry points construct the
kernel once per call and pass it to the per-state solvers.
"""
_build_kernel(cdp::ContinuousDP, s_probe, x_probe) =
    _build_kernel_w(cdp.weights, cdp, s_probe, x_probe)

_build_kernel_w(w::AbstractVector, cdp::ContinuousDP, s_probe, x_probe) =
    _QuadratureKernel(cdp.g, cdp.shocks, w)

function _build_kernel_w(w, cdp::ContinuousDP, s_probe, x_probe)
    wrapped = if hasmethod(w, Tuple{typeof(s_probe),typeof(x_probe)})
        _StateActionWeights(w)
    elseif hasmethod(w, Tuple{typeof(s_probe)})
        _StateWeights(w)
    else
        throw(ArgumentError(
            "callable `weights` must accept a state `weights(s)` or a " *
            "state-action pair `weights(s, x)`"))
    end
    probe = _fetch_weights(wrapped, s_probe, x_probe)
    _check_weights_probe(probe, size(cdp.shocks, 1))
    return _QuadratureKernel(cdp.g, cdp.shocks, wrapped)
end

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

# Draw a branch index from the branch distribution at (s, x) by inverse
# CDF, accumulating the weights without materializing them
function _draw_branch_index(rng::AbstractRNG, ker::_QuadratureKernel, s, x)
    w = _branch_weights(ker, s, x)
    r = rand(rng)
    acc = 0.0
    last = 1
    for j in eachindex(w)
        acc += w[j]
        last = j
        r <= acc && return j
    end
    return last  # guard against roundoff when the weights sum to ~1
end
