"""
    _QuadratureKernel{TG,TR}

Internal representation of the transition kernel of a `ContinuousDP`: the
distribution of the next state given `(s, x)`, as finitely many weighted
branches `(s'_j, w_j)`. This structured kernel pairs the transition
function `g` with fixed quadrature nodes and weights; branch `j` has next
state `g(s, x, shocks[j])` and weight `weights[j]`.

Consumers iterate branches by index: fetch the weight container once per
`(s, x)` with `_branch_weights` and evaluate next states with
`_branch_state`; derivative-based consumers re-evaluate `_branch_state`
at perturbed actions holding the branch index fixed. The plain expected
value of an interpolant is provided by `_expected_value`.
"""
struct _QuadratureKernel{TG,TR<:AbstractVecOrMat}
    g::TG
    shocks::TR
    weights::Vector{Float64}
end

# Construction is free (a non-escaping bundle of references): callers may
# construct per call rather than threading a kernel through signatures.
_kernel(cdp::ContinuousDP) = _QuadratureKernel(cdp.g, cdp.shocks, cdp.weights)

# Branch weights at (s, x), as an indexable collection aligned with the
# branch indices. Fixed quadrature weights ignore (s, x).
_branch_weights(ker::_QuadratureKernel, s, x) = ker.weights

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
