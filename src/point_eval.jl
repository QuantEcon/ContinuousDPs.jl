#=
Non-allocating point evaluation of interpolants on BasisMatrices.jl bases.

Evaluate a fitted function with basis coefficients `C` at a single point
without constructing `BasisMatrix` objects, using a small preallocated
workspace. Intended to be lifted upstream to BasisMatrices.jl; this file is
deliberately self-contained and free of any types defined in ContinuousDPs.

The basis values are computed with exactly the same conventions as
`BasisMatrices.evalbase`, including the behavior for points outside the
interpolation domain:

* `ChebParams`: Chebyshev recurrence on the unscaled point (polynomial
  extrapolation outside `[a, b]`).
* `SplineParams`: de Boor recurrence on the augmented breakpoint sequence,
  with the point assigned to the first/last interval outside the domain
  (i.e., replicating `lookup(augbreaks, x, 3)`).
* `LinParams`: piecewise linear weights, extrapolated linearly outside the
  domain.

The kernels work in `Float64` internally: evaluation points are accepted as
`Real` and converted to `Float64`, while basis coefficients are required to
be `Float64` (so that unintended use with other numeric types, e.g.
`BigFloat` or AD dual numbers, fails at dispatch instead of silently losing
precision).

Derivatives of the interpolant are supported through coefficient
differentiation ([`DerivFunEvalCache`](@ref)): for each basis family, the
derivative of an interpolant is itself an interpolant on the differentiated
parameters (`derivative_op` from BasisMatrices.jl provides both the sparse
coefficient operator and the differentiated parameters), so a partial
derivative is evaluated with the same order-0 kernels after a single sparse
transformation of the coefficient vector.
=#
using BasisMatrices: BasisParams, ChebParams, SplineParams, LinParams, Basis,
                     evalbase, derivative_op
using LinearAlgebra: I, mul!
using SparseArrays: SparseMatrixCSC, sparse

#= Per-dimension evaluation caches =#

"""
    PointEvalCache

Abstract type for one-dimensional basis evaluation caches. A concrete cache
holds the basis parameters together with a preallocated buffer `vals` for the
basis function values at a point. See [`point_evalbase!`](@ref).
"""
abstract type PointEvalCache end

struct ChebEvalCache{TP<:ChebParams} <: PointEvalCache
    p::TP
    vals::Vector{Float64}
end

struct SplineEvalCache{TP<:SplineParams} <: PointEvalCache
    p::TP
    augbreaks::Vector{Float64}
    numfirst::Int  # lower-endpoint index returned by `lookup(augbreaks, x, 3)`
    upper::Int     # upper-endpoint index returned by `lookup(augbreaks, x, 3)`
    vals::Vector{Float64}
end

struct LinEvalCache{TP<:LinParams} <: PointEvalCache
    p::TP
    vals::Vector{Float64}
end

# Fallback for basis families without a specialized kernel. Correct but slow
# (allocates): calls `evalbase` with a one-point array.
struct GenericEvalCache{TP<:BasisParams} <: PointEvalCache
    p::TP
    vals::Vector{Float64}
end

"""
    PointEvalCache(p::BasisParams)

Construct an evaluation cache for the one-dimensional basis described by `p`.
Specialized non-allocating kernels are provided for `ChebParams`,
`SplineParams`, and `LinParams`; other parameter types fall back to a generic
(allocating) implementation based on `evalbase`.
"""
PointEvalCache(p::ChebParams) =
    ChebEvalCache(p, Vector{Float64}(undef, p.n))

function PointEvalCache(p::SplineParams)
    k, breaks = p.k, p.breaks
    augbreaks = convert(
        Vector{Float64},
        vcat(fill(breaks[1], k), collect(breaks), fill(breaks[end], k))
    )
    m = length(augbreaks)
    numfirst = 1
    while numfirst < m && augbreaks[numfirst+1] == augbreaks[1]
        numfirst += 1
    end
    upper = m - 1
    while upper >= 1 && augbreaks[upper] == augbreaks[m]
        upper -= 1
    end
    return SplineEvalCache(p, augbreaks, numfirst, upper,
                           Vector{Float64}(undef, k+1))
end

PointEvalCache(p::LinParams) = LinEvalCache(p, Vector{Float64}(undef, 2))

PointEvalCache(p::BasisParams) =
    GenericEvalCache(p, Vector{Float64}(undef, length(p)))

# Scalar version of `BasisMatrices.lookup(table, x, 3)`; the scans for
# repeated endpoint values run only in the rare out-of-range branches.
@inline function _lookup3(table::AbstractVector, x::Real)
    m = length(table)
    ind = searchsortedfirst(table, x) - 1
    if ind == m
        i = m - 1
        while i >= 1 && table[i] == table[m]
            i -= 1
        end
        return i
    end
    if ind == 0
        i = 1
        while i < m && table[i+1] == table[1]
            i += 1
        end
        return i
    end
    return ind
end

"""
    point_evalbase!(cache::PointEvalCache, x::Real)

Evaluate the basis functions with nonzero support at the point `x`, storing
the values in `cache.vals`.

# Returns

- `first::Int`: Index of the first basis function with nonzero support.
- `nvals::Int`: Number of (potentially) nonzero basis functions;
  `cache.vals[i]` is the value of basis function `first + i - 1` for
  `i in 1:nvals`.
"""
function point_evalbase!(cache::ChebEvalCache, x::Real)
    p = cache.p
    n = p.n
    vals = cache.vals
    z = (2 / (p.b - p.a)) * (x - (p.a + p.b) / 2)
    @inbounds begin
        vals[1] = 1.0
        n >= 2 && (vals[2] = z)
        z2 = 2 * z
        for j in 3:n
            vals[j] = z2 * vals[j-1] - vals[j-2]
        end
    end
    return 1, n
end

function point_evalbase!(cache::SplineEvalCache, x::Real)
    k = cache.p.k
    aug = cache.augbreaks
    vals = cache.vals

    # Interval index, replicating `lookup(augbreaks, x, 3)`
    ind = searchsortedfirst(aug, x) - 1
    if ind == length(aug)
        ind = cache.upper
    elseif ind == 0
        ind = cache.numfirst
    end

    # de Boor recurrence, as in `evalbase(::SplineParams, ...)`
    @inbounds begin
        vals[1] = 1.0
        for j in 2:k+1
            vals[j] = 0.0
        end
        for j in 1:k
            for jj in j:-1:1
                b0 = aug[ind+jj-j]
                b1 = aug[ind+jj]
                temp = vals[jj] / (b1 - b0)
                vals[jj+1] = (x - b0) * temp + vals[jj+1]
                vals[jj] = (b1 - x) * temp
            end
        end
    end
    return ind - k, k + 1
end

function point_evalbase!(cache::LinEvalCache, x::Real)
    p = cache.p
    breaks = p.breaks
    n = length(breaks)
    if p.evennum != 0
        step_inv = (n - 1) / (breaks[end] - breaks[1])
        ind = clamp(trunc(Int, (x - breaks[1]) * step_inv) + 1, 1, n-1)
    else
        ind = _lookup3(breaks, x)
    end
    @inbounds begin
        z = (x - breaks[ind]) / (breaks[ind+1] - breaks[ind])
        cache.vals[1] = 1 - z
        cache.vals[2] = z
    end
    return ind, 2
end

function point_evalbase!(cache::GenericEvalCache, x::Real)
    B = evalbase(cache.p, [x], 0)
    n = length(cache.vals)
    for j in 1:n
        cache.vals[j] = B[1, j]
    end
    return 1, n
end


#= N-dimensional interpolant evaluation =#

"""
    FunEvalCache{N,TE}

Workspace for evaluating an interpolant on an `N`-dimensional tensor-product
`Basis` at a single point without allocations. Construct with
`FunEvalCache(basis)` and evaluate with [`funeval_point!`](@ref).

Not thread-safe: use one cache per thread.
"""
struct FunEvalCache{N,TE<:NTuple{N,PointEvalCache}}
    caches::TE
    dims::NTuple{N,Int}
    strides::NTuple{N,Int}
end

"""
    FunEvalCache(basis::Basis{N})

Construct a point-evaluation workspace for `basis`.
"""
function FunEvalCache(basis::Basis{N}) where N
    caches = ntuple(d -> PointEvalCache(basis.params[d]), Val(N))
    dims = ntuple(d -> length(basis.params[d]), Val(N))
    strides = ntuple(d -> prod(dims[1:d-1]), Val(N))
    return FunEvalCache(caches, dims, strides)
end

@inline _coord(x::Real, d::Int) = x
@inline _coord(x, d::Int) = @inbounds x[d]

"""
    funeval_point!(fec::FunEvalCache{N}, C, x) -> Float64

Evaluate the interpolant with basis coefficients `C` at the single point `x`,
using (and overwriting) the workspace `fec`. Equivalent to
`funeval(C, basis, x)` from BasisMatrices.jl, but non-allocating.

# Arguments

- `fec::FunEvalCache{N}`: Workspace constructed from the same `Basis` that
  `C` was fitted on.
- `C::AbstractVector{Float64}`: Basis coefficient vector, in the same
  (expanded) ordering used by `funeval`.
- `x`: Evaluation point: a `Real` if `N == 1`, otherwise anything indexable
  of length `N` (e.g. `Tuple`, `AbstractVector`).

# Returns

- `::Float64`: Value of the interpolant at `x`.
"""
function funeval_point!(fec::FunEvalCache{N}, C::AbstractVector{Float64},
                        x) where N
    firsts_nvals = ntuple(
        d -> point_evalbase!(fec.caches[d], _coord(x, d)), Val(N)
    )
    firsts = map(first, firsts_nvals)
    nvals = map(last, firsts_nvals)
    valvecs = ntuple(d -> fec.caches[d].vals, Val(N))
    strides = fec.strides

    base = 1
    for d in 1:N
        base += (firsts[d] - 1) * strides[d]
    end

    vals1 = valvecs[1]
    nvals1 = nvals[1]
    acc = 0.0
    @inbounds for jrest in CartesianIndices(Base.tail(nvals))
        w = 1.0
        offset = base
        for d in 2:N
            w *= valvecs[d][jrest[d-1]]
            offset += (jrest[d-1] - 1) * strides[d]
        end
        acc1 = 0.0
        @simd for i in 0:nvals1-1
            acc1 += vals1[i+1] * C[offset+i]
        end
        acc += w * acc1
    end
    return acc
end


#= Point evaluation of derivatives =#

# Differentiated parameters and the composed 1-D coefficient operator for a
# derivative of order `o` (identity for o == 0)
function _deriv_params_op(p::BasisParams, o::Int)
    o == 0 && return p, sparse(1.0I, length(p), length(p))
    D, p_new = derivative_op(p, [0.0], o)
    return p_new, SparseMatrixCSC{Float64,Int}(D[o])
end

function _deriv_params_op(p::SplineParams, o::Int)
    o == 0 && return p, sparse(1.0I, length(p), length(p))
    # `evalbase` requires the derivative order to be less than the spline
    # degree; mirror that restriction here
    o < p.k || throw(ArgumentError(
        "derivative order ($o) must be less than the spline degree ($(p.k))"))
    D, p_new = derivative_op(p, [0.0], o)
    return p_new, SparseMatrixCSC{Float64,Int}(D[o])
end

"""
    DerivFunEvalCache{N,TF}

Workspace for evaluating a partial derivative of an interpolant on an
`N`-dimensional tensor-product `Basis` at single points without allocations.

The derivative of the interpolant is represented as an interpolant on the
differentiated basis parameters, whose coefficients are a (sparse) linear
transformation of the original ones. Set the coefficients with
[`set_coefs!`](@ref) (once for each new coefficient vector), then evaluate
with `funeval_point!(dfec, x)` at any number of points.

Not thread-safe: use one cache per thread.
"""
struct DerivFunEvalCache{N,TF<:FunEvalCache{N}}
    order::NTuple{N,Int}
    fec::TF
    D::SparseMatrixCSC{Float64,Int}
    C_deriv::Vector{Float64}
end

"""
    DerivFunEvalCache(basis::Basis{N}, order::NTuple{N,Int})

Construct a point-evaluation workspace for the partial derivative of orders
`order` (one nonnegative integer per dimension, `0` meaning no
differentiation) of interpolants on `basis`. For `SplineParams`, the order
must be less than the spline degree, as in `evalbase`.
"""
function DerivFunEvalCache(basis::Basis{N}, order::NTuple{N,Int}) where N
    all(o -> o >= 0, order) ||
        throw(ArgumentError("derivative orders must be nonnegative"))
    params_ops = ntuple(d -> _deriv_params_op(basis.params[d], order[d]),
                        Val(N))
    fec = FunEvalCache(Basis(map(first, params_ops)))
    D = reduce(kron, (params_ops[d][2] for d in N:-1:1))
    C_deriv = Vector{Float64}(undef, size(D, 1))
    return DerivFunEvalCache(order, fec, D, C_deriv)
end

"""
    set_coefs!(dfec::DerivFunEvalCache, C)

Set the basis coefficients of the interpolant to differentiate: transform
`C` into the coefficients of the derivative interpolant, stored in `dfec`.
Call once whenever `C` changes, before evaluating with `funeval_point!`.
"""
function set_coefs!(dfec::DerivFunEvalCache, C::AbstractVector{Float64})
    mul!(dfec.C_deriv, dfec.D, C)
    return dfec
end

"""
    funeval_point!(dfec::DerivFunEvalCache, x) -> Float64

Evaluate the partial derivative of the interpolant whose coefficients were
set by [`set_coefs!`](@ref) at the single point `x`. Equivalent to `funeval`
with the corresponding derivative `order`, but non-allocating.
"""
funeval_point!(dfec::DerivFunEvalCache, x) =
    funeval_point!(dfec.fec, dfec.C_deriv, x)
