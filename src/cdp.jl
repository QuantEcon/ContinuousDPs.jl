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


"""
    ContinuousDP{N,Tf,Tg,TR,Tlb,Tub,TI}

Type representing a continuous-state dynamic program with `N`-dimensional state
space.

# Fields

- `f::Tf`: Reward function `f(s, x)`.
- `g::Tg`: State transition function `g(s, x, e)`.
- `discount::Float64`: Discount factor.
- `shocks::TR<:AbstractVecOrMat`: Discretized shock nodes.
- `weights::Vector{Float64}`: Probability weights for the shock nodes.
- `x_lb::Tlb`: Lower bound of the action variable as a function of state.
- `x_ub::Tub`: Upper bound of the action variable as a function of state.
- `interp::TI<:Interp{N}`: Object that contains the information about the
  interpolation scheme.
"""
struct ContinuousDP{N,Tf,Tg,TR<:AbstractVecOrMat,Tlb,Tub,TI<:Interp{N}}
    f::Tf
    g::Tg
    discount::Float64
    shocks::TR
    weights::Vector{Float64}
    x_lb::Tlb
    x_ub::Tub
    interp::TI
end

"""
    ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, basis)

Constructor for `ContinuousDP`.

# Arguments

- `f`: Reward function `f(s, x)`.
- `g`: State transition function `g(s, x, e)`.
- `discount::Real`: Discount factor.
- `shocks::AbstractVecOrMat`: Discretized shock nodes.
- `weights::Vector{Float64}`: Probability weights for the shock nodes.
- `x_lb`: Lower bound of the action variable as a function of state.
- `x_ub`: Upper bound of the action variable as a function of state.
- `basis::Basis`: Object that contains the interpolation basis information.
"""
function ContinuousDP(f, g, discount::Real,
                      shocks::AbstractVecOrMat, weights::Vector{Float64},
                      x_lb, x_ub,
                      basis::Basis{N}) where {N}
    interp = Interp(basis)
    cdp = ContinuousDP(f, g, Float64(discount), shocks, weights, x_lb, x_ub, interp)
    return cdp
end

"""
    ContinuousDP(cdp::ContinuousDP; f=cdp.f, g=cdp.g, discount=cdp.discount,
                 shocks=cdp.shocks, weights=cdp.weights,
                 x_lb=cdp.x_lb, x_ub=cdp.x_ub, basis=cdp.interp.basis)

Construct a copy of `cdp`, optionally replacing selected model components.
"""
function ContinuousDP(cdp::ContinuousDP;
    f = cdp.f,
    g = cdp.g,
    discount = cdp.discount,
    shocks = cdp.shocks,
    weights = cdp.weights,
    x_lb = cdp.x_lb,
    x_ub = cdp.x_ub,
    basis = cdp.interp.basis
)
    return ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, basis)
end


"""
    CDPSolveResult{Algo,N,TCDP,TE}

Type storing the solution of a continuous-state dynamic program obtained by
algorithm `Algo`.

# Fields

- `cdp::TCDP<:ContinuousDP{N}`: The dynamic program that was solved.
- `tol::Float64`: Convergence tolerance used by the solver.
- `max_iter::Int`: Maximum number of iterations allowed.
- `C::Vector{Float64}`: Basis coefficient vector for the fitted value function.
- `converged::Bool`: Whether the algorithm converged.
- `num_iter::Int`: Number of iterations performed.
- `eval_nodes::TE<:VecOrMat`: Nodes at which the solution is evaluated.
  Defaults to `cdp.interp.S`.
- `eval_nodes_coord::NTuple{N,Vector{Float64}}`: Coordinate vectors of the
  evaluation nodes along each dimension. Defaults to `cdp.interp.Scoord`.
- `V::Vector{Float64}`: Value function evaluated at `eval_nodes`.
- `X::Vector{Float64}`: Policy function evaluated at `eval_nodes`.
- `resid::Vector{Float64}`: Approximation residuals at `eval_nodes`.
"""
mutable struct CDPSolveResult{Algo<:DPAlgorithm,N,TCDP<:ContinuousDP{N},
                              TE<:VecOrMat}
    cdp::TCDP
    tol::Float64
    max_iter::Int
    C::Vector{Float64}
    converged::Bool
    num_iter::Int
    eval_nodes::TE
    eval_nodes_coord::NTuple{N,Vector{Float64}}
    V::Vector{Float64}
    X::Vector{Float64}
    resid::Vector{Float64}

    function CDPSolveResult{Algo,N}(
            cdp::TCDP, tol::Float64, max_iter::Integer
        ) where {Algo,N,TCDP<:ContinuousDP{N}}
        C = zeros(cdp.interp.length)
        converged = false
        num_iter = 0
        eval_nodes = cdp.interp.S
        eval_nodes_coord = cdp.interp.Scoord
        V = Float64[]
        X = Float64[]
        resid = Float64[]
        res = new{Algo,N,TCDP,typeof(eval_nodes)}(
            cdp, tol, max_iter, C, converged, num_iter,
            eval_nodes, eval_nodes_coord, V, X, resid
        )
        return res
    end
end

Base.ndims(::ContinuousDP{N}) where {N} = N
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
    n = size(s_nodes, 1)
    length(res.V) == n || (res.V = Vector{Float64}(undef, n))
    length(res.X) == n || (res.X = Vector{Float64}(undef, n))
    length(res.resid) == n || (res.resid = Vector{Float64}(undef, n))
    s_wise_max!(cdp, s_nodes, C, res.V, res.X, fec)
    for i in 1:n
        res.resid[i] = res.V[i] - funeval_point!(fec, C, _row(s_nodes, i))
    end
    return res
end

evaluate!(res::CDPSolveResult) =
    evaluate!(res, FunEvalCache(res.cdp.interp.basis))

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
    fec = FunEvalCache(cdp.interp.basis)
    V, X = s_wise_max(cdp, s_nodes, C, fec)
    n = size(s_nodes, 1)
    resid = Vector{Float64}(undef, n)
    for i in 1:n
        resid[i] = V[i] - funeval_point!(fec, C, _row(s_nodes, i))
    end
    return V, X, resid
end


"""
    CDPWorkspace{TF,TD}

Preallocated buffers used by the solution algorithms for a `ContinuousDP`.
Construct with `CDPWorkspace(cdp; inner_solver=:foc)`.

Not thread-safe: use one workspace per thread.

# Fields

- `fec::TF<:FunEvalCache`: Workspace for point evaluation of the value
  function.
- `dfecs::TD`: Tuple of `DerivFunEvalCache`s for the gradient of the value
  function (one per state dimension), or `nothing` if the basis does not
  support the first-order-condition solver (see below).
- `Tv::Vector{Float64}`: Buffer for updated values at the interpolation
  nodes.
- `X::Vector{Float64}`: Buffer for updated actions at the interpolation
  nodes (initialized to `NaN`; also serves as the warm start for the inner
  maximization in the next sweep).
- `inner_solver::Symbol`: `:foc` to solve the inner maximization via its
  first-order condition (with Brent as automatic fallback), or `:brent` to
  always use derivative-free Brent maximization.
"""
struct CDPWorkspace{TF<:FunEvalCache,TD}
    fec::TF
    dfecs::TD
    Tv::Vector{Float64}
    X::Vector{Float64}
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
    CDPWorkspace(cdp::ContinuousDP; inner_solver=:foc)

Construct a `CDPWorkspace` for `cdp`. See [`solve`](@ref) for the meaning of
`inner_solver`.
"""
function CDPWorkspace(cdp::ContinuousDP{N}; inner_solver::Symbol=:foc) where N
    inner_solver in (:foc, :brent) ||
        throw(ArgumentError("inner_solver must be :foc or :brent"))
    basis = cdp.interp.basis
    dfecs = if inner_solver == :foc &&
               all(d -> _foc_suitable(basis.params[d]), 1:N)
        ntuple(d -> DerivFunEvalCache(
                   basis, ntuple(i -> Int(i == d), Val(N))), Val(N))
    else
        nothing
    end
    return CDPWorkspace(
        FunEvalCache(basis),
        dfecs,
        Vector{Float64}(undef, cdp.interp.length),
        fill(NaN, cdp.interp.length),
        inner_solver
    )
end


#= Methods =#

# Non-copying access to the i-th point of a set of points stored as a
# Vector (one point = one scalar) or a Matrix (one point = one row).
_row(A::AbstractVector, i::Int) = A[i]
_row(A::AbstractMatrix, i::Int) = view(A, i, :)

"""
    _s_wise_max!(cdp, s, C, fec)

Find the optimal value and action at a given state `s`.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `s`: State point at which to maximize.
- `C`: Basis coefficient vector for the value function.
- `fec::FunEvalCache`: Workspace for evaluating the value function at the
  next states.

# Returns

- `v::Float64`: Optimal value at `s`.
- `x::Float64`: Optimal action at `s`.
"""
function _s_wise_max!(cdp::ContinuousDP, s, C, fec::FunEvalCache)
    function objective(x)
        cont = 0.0
        for j in eachindex(cdp.weights)
            e = _row(cdp.shocks, j)
            s_next = cdp.g(s, x, e)
            cont += cdp.weights[j] * funeval_point!(fec, C, s_next)
        end
        flow = cdp.f(s, x)
        return flow + cdp.discount * cont
    end
    res = Optim.maximize(objective, cdp.x_lb(s), cdp.x_ub(s))
    v = Optim.maximum(res)::Float64
    x = Optim.maximizer(res)::Float64
    return v, x
end


#= First-order-condition inner solver =#

# Relative step for central finite differences of the user's f and g
const _FOC_RELSTEP = cbrt(eps(Float64))

"""
    _objective_and_deriv(cdp, s, C, fec, dfecs, x, x_lb, x_ub)

Evaluate the inner objective `H(x) = f(s, x) + beta * E[V^(g(s, x, e))]` and
its derivative `H'(x) = f_x + beta * E[grad V^(g(s, x, e)) . g_x]` at the
action `x`. The gradient of the fitted value function is evaluated exactly
via `dfecs` (whose coefficients must be set with `set_coefs!` beforehand),
while `f_x` and `g_x` are computed by central finite differences with the
step shrunk so that `x ± h` stay within `[x_lb, x_ub]` (so that `f` and `g`
are never called at infeasible actions); if no adequate step exists,
`(NaN, NaN)` is returned to trigger the Brent fallback.

# Returns

- `H::Float64`, `Hp::Float64`: Objective value and derivative (may be
  non-finite, in which case the caller should fall back to Brent).
"""
function _objective_and_deriv(cdp::ContinuousDP{N}, s, C, fec::FunEvalCache,
                              dfecs, x::Float64, x_lb::Float64,
                              x_ub::Float64) where N
    h = min(_FOC_RELSTEP * max(abs(x), 1.0), x - x_lb, x_ub - x)
    h > eps() * max(abs(x), 1.0) || return NaN, NaN
    f0 = cdp.f(s, x)
    fp = (cdp.f(s, x + h) - cdp.f(s, x - h)) / (2h)
    cont = 0.0
    contp = 0.0
    for j in eachindex(cdp.weights)
        e = _row(cdp.shocks, j)
        s_next = cdp.g(s, x, e)
        s_up = cdp.g(s, x + h, e)
        s_dn = cdp.g(s, x - h, e)
        v = funeval_point!(fec, C, s_next)
        dv = _grad_dot_gx(dfecs, s_next, s_up, s_dn, h)
        w = cdp.weights[j]
        cont += w * v
        contp += w * dv
    end
    H = f0 + cdp.discount * cont
    Hp = fp + cdp.discount * contp
    return H, Hp
end

# grad V^(s_next) . g_x, with g_x by central differences, unrolled over the
# state dimensions
@inline function _grad_dot_gx(dfecs::NTuple{N,DerivFunEvalCache}, s_next,
                              s_up, s_dn, h::Float64) where N
    parts = ntuple(
        d -> funeval_point!(dfecs[d], s_next) *
             ((_coord(s_up, d) - _coord(s_dn, d)) / (2h)),
        Val(N)
    )
    return sum(parts)
end

"""
    _s_wise_max_foc!(cdp, s, C, fec, dfecs, x_prev)

Find the optimal value and action at state `s` by solving the first-order
condition `H'(x) = 0` with safeguarded bracketing root-finding (regula falsi
with the Illinois modification), warm-started at `x_prev` (`NaN` for a cold
start). Falls back to the Brent-based `_s_wise_max!` whenever the objective
or its derivative is non-finite at a required point. Corner solutions are
detected from the sign of `H'` during the bracketing expansion.

The coefficients of `dfecs` must have been set with `set_coefs!(., C)`.
"""
function _s_wise_max_foc!(cdp::ContinuousDP, s, C, fec::FunEvalCache, dfecs,
                          x_prev::Float64)
    lb, ub = Float64(cdp.x_lb(s)), Float64(cdp.x_ub(s))
    width = ub - lb
    width > 0 || return _s_wise_max!(cdp, s, C, fec)

    # Evaluation points are kept slightly inside the bounds, where f is more
    # likely to be finite (e.g. log(x) at x_lb = 0)
    off = sqrt(eps()) * width
    lo, hi = lb + off, ub - off

    x0 = isfinite(x_prev) ? clamp(x_prev, lo, hi) : 0.5 * (lo + hi)
    H0, Hp0 = _objective_and_deriv(cdp, s, C, fec, dfecs, x0, lb, ub)
    (isfinite(H0) && isfinite(Hp0)) ||
        return _s_wise_max!(cdp, s, C, fec)

    # Bracket a sign change of H' by expanding from x0 in the uphill
    # direction; `a` keeps H' > 0, `b` keeps H' < 0
    a, Ha, Hpa = x0, H0, Hp0
    b, Hb, Hpb = x0, H0, Hp0
    if Hp0 > 0
        step = 0.02 * width
        while true
            xt = min(a + step, hi)
            Ht, Hpt = _objective_and_deriv(cdp, s, C, fec, dfecs, xt, lb, ub)
            (isfinite(Ht) && isfinite(Hpt)) ||
                return _s_wise_max!(cdp, s, C, fec)
            if Hpt <= 0
                b, Hb, Hpb = xt, Ht, Hpt
                break
            end
            a, Ha, Hpa = xt, Ht, Hpt
            xt >= hi && return Ht, xt  # H increasing up to the bound
            step *= 4
        end
    elseif Hp0 < 0
        step = 0.02 * width
        while true
            xt = max(b - step, lo)
            Ht, Hpt = _objective_and_deriv(cdp, s, C, fec, dfecs, xt, lb, ub)
            (isfinite(Ht) && isfinite(Hpt)) ||
                return _s_wise_max!(cdp, s, C, fec)
            if Hpt >= 0
                a, Ha, Hpa = xt, Ht, Hpt
                break
            end
            b, Hb, Hpb = xt, Ht, Hpt
            xt <= lo && return Ht, xt  # H decreasing down from the bound
            step *= 4
        end
    else  # Hp0 == 0: already at a stationary point
        return H0, x0
    end

    # Safeguarded root-finding on H' over [a, b] with H'(a) > 0 > H'(b):
    # regula falsi with the Illinois modification, bisection safeguard
    wa, wb = Hpa, Hpb  # (possibly rescaled) values used for interpolation
    side = 0
    xtol = sqrt(eps()) * max(1.0, abs(a), abs(b))
    for _ in 1:60
        b - a <= xtol && break
        xm = (a * wb - b * wa) / (wb - wa)
        if !(a < xm < b)
            xm = 0.5 * (a + b)
        end
        Hm, Hpm = _objective_and_deriv(cdp, s, C, fec, dfecs, xm, lb, ub)
        (isfinite(Hm) && isfinite(Hpm)) ||
            return _s_wise_max!(cdp, s, C, fec)
        if Hpm > 0
            a, Ha = xm, Hm
            wa = Hpm
            side == 1 && (wb *= 0.5)
            side = 1
        elseif Hpm < 0
            b, Hb = xm, Hm
            wb = Hpm
            side == -1 && (wa *= 0.5)
            side = -1
        else
            return Hm, xm
        end
    end
    return Ha >= Hb ? (Ha, a) : (Hb, b)
end

"""
    _s_wise_max_foc_sweep!(cdp, C, Tv, X, fec, dfecs)

Run the FOC-based inner maximization over all interpolation nodes, storing
values in `Tv` and maximizers in `X`. The previous contents of `X` serve as
warm starts (`NaN` entries mean cold start). Sets the coefficients of
`dfecs` from `C`. Falls back to Brent state-by-state on exceptions from the
model functions (e.g. a `DomainError` at a finite-difference point).
"""
function _s_wise_max_foc_sweep!(cdp::ContinuousDP, C::Vector{Float64},
                                Tv::Vector{Float64}, X::Vector{Float64},
                                fec::FunEvalCache, dfecs)
    foreach(dfec -> set_coefs!(dfec, C), dfecs)
    ss = cdp.interp.S
    for i in 1:size(ss, 1)
        s = _row(ss, i)
        Tv[i], X[i] =
            try
                _s_wise_max_foc!(cdp, s, C, fec, dfecs, X[i])
            catch err
                err isa InterruptException && rethrow()
                _s_wise_max!(cdp, s, C, fec)
            end
    end
    return Tv, X
end

_use_foc(ws::CDPWorkspace) = ws.inner_solver == :foc && ws.dfecs !== nothing

"""
    s_wise_max!(cdp, ss, C, Tv[, fec])

Find optimal value for each grid point.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `Tv::Vector{Float64}`: A buffer array to hold the updated value function.
- `fec::FunEvalCache`: Workspace for point evaluation of the value function.
  Constructed internally if not given.

# Returns

- `Tv::Vector{Float64}`: Updated value function vector.
"""
function s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64},
                     fec::FunEvalCache)
    n = size(ss, 1)
    for i in 1:n
        Tv[i], _ = _s_wise_max!(cdp, _row(ss, i), C, fec)
    end
    return Tv
end

s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
            C::Vector{Float64}, Tv::Vector{Float64}) =
    s_wise_max!(cdp, ss, C, Tv, FunEvalCache(cdp.interp.basis))

"""
    s_wise_max!(cdp, ss, C, Tv, X[, fec])

Find optimal value and action for each grid point.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
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
function s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64},
                     X::Vector{Float64}, fec::FunEvalCache)
    n = size(ss, 1)
    for i in 1:n
        Tv[i], X[i] = _s_wise_max!(cdp, _row(ss, i), C, fec)
    end
    return Tv, X
end

s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
            C::Vector{Float64}, Tv::Vector{Float64}, X::Vector{Float64}) =
    s_wise_max!(cdp, ss, C, Tv, X, FunEvalCache(cdp.interp.basis))

"""
    s_wise_max(cdp, ss, C[, fec])

Find optimal value and action for each grid point.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `fec::FunEvalCache`: Workspace for point evaluation of the value function.
  Constructed internally if not given.

# Returns

- `Tv::Vector{Float64}`: Value function vector.
- `X::Vector{Float64}`: Policy function vector.
"""
function s_wise_max(cdp::ContinuousDP, ss::AbstractArray{Float64},
                    C::Vector{Float64}, fec::FunEvalCache)
    n = size(ss, 1)
    Tv, X = Array{Float64}(undef, n), Array{Float64}(undef, n)
    s_wise_max!(cdp, ss, C, Tv, X, fec)
end

s_wise_max(cdp::ContinuousDP, ss::AbstractArray{Float64},
           C::Vector{Float64}) =
    s_wise_max(cdp, ss, C, FunEvalCache(cdp.interp.basis))


"""
    bellman_operator!(cdp, C, Tv)
    bellman_operator!(cdp, C, ws)

Apply the Bellman operator and update the basis coefficients. Values are
stored in `Tv` (or `ws.Tv`).

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `Tv::Vector{Float64}`: Vector to store values.
- `ws::CDPWorkspace`: Workspace for the solution algorithms.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function bellman_operator!(cdp::ContinuousDP, C::Vector{Float64},
                           ws::CDPWorkspace)
    if _use_foc(ws)
        _s_wise_max_foc_sweep!(cdp, C, ws.Tv, ws.X, ws.fec, ws.dfecs)
    else
        s_wise_max!(cdp, cdp.interp.S, C, ws.Tv, ws.X, ws.fec)
    end
    ldiv!(C, cdp.interp.Phi_lu, ws.Tv)
    return C
end

function bellman_operator!(cdp::ContinuousDP, C::Vector{Float64},
                           Tv::Vector{Float64})
    Tv = s_wise_max!(cdp, cdp.interp.S, C, Tv)
    ldiv!(C, cdp.interp.Phi_lu, Tv)
    return C
end


"""
    compute_greedy!(cdp, C, X)
    compute_greedy!(cdp, ss, C, X[, fec])

Compute the greedy policy for the given basis coefficients.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.
- `fec::FunEvalCache`: Workspace for point evaluation of the value function.
  Constructed internally if not given.

# Returns

- `X::Vector{Float64}`: Updated policy function vector.
"""
function compute_greedy!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                         C::Vector{Float64}, X::Vector{Float64},
                         fec::FunEvalCache)
    n = size(ss, 1)
    for i in 1:n
        _, X[i] = _s_wise_max!(cdp, _row(ss, i), C, fec)
    end
    return X
end

compute_greedy!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                C::Vector{Float64}, X::Vector{Float64}) =
    compute_greedy!(cdp, ss, C, X, FunEvalCache(cdp.interp.basis))

compute_greedy!(cdp::ContinuousDP, C::Vector{Float64}, X::Vector{Float64}) =
    compute_greedy!(cdp, cdp.interp.S, C, X)

"""
    evaluate_policy!(cdp, X, C[, fec])

Compute the value function for a given policy and update the basis
coefficients: solve `(Phi - beta * E[Phi(g(S, X, e))]) C = f(S, X)`, where
the expected-basis matrix is assembled row by row with the non-allocating
point-evaluation kernels. When the collocation matrix `Phi` is sparse
(spline and piecewise linear bases), the system matrix is assembled and
factorized in sparse form.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `X::Vector{Float64}`: Policy function vector.
- `C::Vector{Float64}`: A buffer array to hold the basis coefficients.
- `fec::FunEvalCache`: Workspace whose per-dimension caches are used for
  basis evaluation at the next states. Constructed internally if not given.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function evaluate_policy!(cdp::ContinuousDP, X::Vector{Float64},
                          C::Vector{Float64}, fec::FunEvalCache)
    A_lu = _policy_system_lu(cdp.interp.Phi, cdp, X, fec)
    ss = cdp.interp.S
    for i in 1:size(ss, 1)
        C[i] = cdp.f(_row(ss, i), X[i])
    end
    ldiv!(A_lu, C)
    return C
end

evaluate_policy!(cdp::ContinuousDP, X::Vector{Float64},
                 C::Vector{Float64}) =
    evaluate_policy!(cdp, X, C, FunEvalCache(cdp.interp.basis))

# Evaluate the per-dimension basis functions at the point `x` (values are
# left in `fec.caches[d].vals`) and return `(nvals, base)`: the number of
# nonzero basis functions per dimension, and the linear column index of the
# tensor-product basis function formed by the first nonzero function of
# every dimension
@inline function _basis_row_parts(fec::FunEvalCache{N}, x) where N
    firsts_nvals = ntuple(
        d -> point_evalbase!(fec.caches[d], _coord(x, d)), Val(N)
    )
    firsts = map(first, firsts_nvals)
    nvals = map(last, firsts_nvals)
    base = 1
    for d in 1:N
        base += (firsts[d] - 1) * fec.strides[d]
    end
    return nvals, base
end

# Dense path: A = Phi - beta * E[Phi(g(S, X, e))] assembled in place,
# factorized with an in-place dense LU
function _policy_system_lu(Phi::AbstractMatrix, cdp::ContinuousDP, X, fec)
    n = size(cdp.interp.S, 1)
    A = copyto!(Matrix{Float64}(undef, n, n), Phi)
    ss = cdp.interp.S
    for i in 1:n
        s = _row(ss, i)
        for j in eachindex(cdp.weights)
            e = _row(cdp.shocks, j)
            s_next = cdp.g(s, X[i], e)
            _sub_basis_row!(A, i, fec, s_next,
                            cdp.discount * cdp.weights[j])
        end
    end
    return lu!(A)
end

# A[i, :] .-= coef * (basis-function values at x), touching only the
# nonzero entries
function _sub_basis_row!(A::Matrix{Float64}, i::Int, fec::FunEvalCache{N},
                         x, coef::Float64) where N
    nvals, base = _basis_row_parts(fec, x)
    valvecs = ntuple(d -> fec.caches[d].vals, Val(N))
    strides = fec.strides
    vals1 = valvecs[1]
    nvals1 = nvals[1]
    @inbounds for jrest in CartesianIndices(Base.tail(nvals))
        w = coef
        offset = base
        for d in 2:N
            w *= valvecs[d][jrest[d-1]]
            offset += (jrest[d-1] - 1) * strides[d]
        end
        for t in 0:nvals1-1
            A[i, offset+t] -= w * vals1[t+1]
        end
    end
    return A
end

# Sparse path: assemble E = E[Phi(g(S, X, e))] in triplet form (exploiting
# that only few basis functions are nonzero at each point), then factorize
# A = Phi - beta * E with a sparse LU
function _policy_system_lu(Phi::SparseMatrixCSC, cdp::ContinuousDP, X, fec)
    n = size(cdp.interp.S, 1)
    Is, Js, Vs = Int[], Int[], Float64[]
    ss = cdp.interp.S
    for i in 1:n
        s = _row(ss, i)
        for j in eachindex(cdp.weights)
            e = _row(cdp.shocks, j)
            s_next = cdp.g(s, X[i], e)
            _append_basis_row!(Is, Js, Vs, i, fec, s_next, cdp.weights[j])
        end
    end
    E = sparse(Is, Js, Vs, n, n)  # sums duplicate entries
    return lu(Phi - cdp.discount * E)
end

# Append coef * (basis-function values at x) to the triplets of row i
function _append_basis_row!(Is::Vector{Int}, Js::Vector{Int},
                            Vs::Vector{Float64}, i::Int,
                            fec::FunEvalCache{N}, x, coef::Float64) where N
    nvals, base = _basis_row_parts(fec, x)
    valvecs = ntuple(d -> fec.caches[d].vals, Val(N))
    strides = fec.strides
    vals1 = valvecs[1]
    nvals1 = nvals[1]
    @inbounds for jrest in CartesianIndices(Base.tail(nvals))
        w = coef
        offset = base
        for d in 2:N
            w *= valvecs[d][jrest[d-1]]
            offset += (jrest[d-1] - 1) * strides[d]
        end
        for t in 0:nvals1-1
            push!(Is, i)
            push!(Js, offset + t)
            push!(Vs, w * vals1[t+1])
        end
    end
    return nothing
end


"""
    policy_iteration_operator!(cdp, C, X)
    policy_iteration_operator!(cdp, C, ws)

Perform one step of policy function iteration and update the basis coefficients.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.
- `ws::CDPWorkspace`: Workspace for the solution algorithms.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function policy_iteration_operator!(cdp::ContinuousDP, C::Vector{Float64},
                                    ws::CDPWorkspace)
    if _use_foc(ws)
        _s_wise_max_foc_sweep!(cdp, C, ws.Tv, ws.X, ws.fec, ws.dfecs)
    else
        compute_greedy!(cdp, cdp.interp.S, C, ws.X, ws.fec)
    end
    evaluate_policy!(cdp, ws.X, C, ws.fec)
    return C
end

function policy_iteration_operator!(cdp::ContinuousDP, C::Vector{Float64},
                                    X::Vector{Float64})
    compute_greedy!(cdp, C, X)
    evaluate_policy!(cdp, X, C)
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


#= Solve methods =#

"""
    solve(cdp, method=PFI; v_init=zeros(cdp.interp.length), tol=sqrt(eps()),
          max_iter=500, verbose=2, print_skip=50, kwargs...)

Solve the continuous-state dynamic program by the specified method.

# Arguments

- `cdp::ContinuousDP`: The dynamic program to solve.
- `method::Type{<:DPAlgorithm}`: Solution method. `VFI` for value
  function iteration, `PFI` for policy function iteration, or `LQA` for linear
  quadratic approximation. Default is `PFI`.
- `v_init::Vector{Float64}`: Initial value function values at interpolation
   nodes.
- `tol::Real`: Convergence tolerance.
- `max_iter::Integer`: Maximum number of iterations.
- `verbose::Integer`: Level of feedback (0 for no output, 1 for warnings only,
  2 for warning and convergence messages during iteration).
- `print_skip::Integer`: If `verbose == 2`, how many iterations between print
  messages.
- `inner_solver::Symbol`: How to solve the inner maximization over actions
  in VFI and PFI. `:foc` (default) solves the first-order condition by
  safeguarded root-finding, warm-started across iterations, using the exact
  gradient of the fitted value function and finite differences of `f` and
  `g`. It is intended for smooth, effectively concave inner problems where
  the first-order condition identifies the maximizing action: it falls
  back to Brent maximization state-by-state when derivative evaluation is
  unavailable or non-finite (and is used only for continuously
  differentiable bases: any Chebyshev, or splines of degree >= 2), but it
  does not attempt to detect nonconcavity or multiple stationary points.
  Use `inner_solver=:brent` for the derivative-free path. Ignored for
  LQA, which has no inner maximization.
- `point::Tuple{ScalarOrArray, ScalarOrArray, ScalarOrArray}`: Keyword argument
  required when `method` is `LQA`. Specify the steady state `(s, x, e)` around
  which the LQ approximation is constructed.

# Returns

- `res::CDPSolveResult`: Solution object of the dynamic program.
"""
function solve(cdp::ContinuousDP{N}, method::Type{Algo}=PFI;
               v_init::Vector{Float64}=zeros(cdp.interp.length),
               tol::Real=sqrt(eps()), max_iter::Integer=500,
               verbose::Integer=2,
               print_skip::Integer=50,
               inner_solver::Symbol=:foc,
               kwargs...) where {Algo<:DPAlgorithm,N}
    tol = Float64(tol)
    res = CDPSolveResult{Algo,N}(cdp, tol, max_iter)
    # LQA has no inner maximization: skip the FOC derivative caches
    ws = CDPWorkspace(cdp; inner_solver=(Algo === LQA ? :brent : inner_solver))
    ldiv!(res.C, cdp.interp.Phi_lu, v_init)
    _solve!(cdp, res, ws, verbose, print_skip; kwargs...)
    evaluate!(res, ws.fec)
    return res
end


# Policy iteration
@doc """
    PFI

Policy function iteration algorithm for `solve`.
"""
PFI

"""
    _solve!(cdp, res, ws, verbose, print_skip)

Implement policy iteration. See `solve` for further details.
"""
function _solve!(cdp::ContinuousDP, res::CDPSolveResult{PFI},
                 ws::CDPWorkspace, verbose, print_skip)
    operator!(C) = policy_iteration_operator!(cdp, C, ws)
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
    _solve!(cdp, res, ws, verbose, print_skip)

Implement value iteration. See `solve` for further details.
"""
function _solve!(cdp::ContinuousDP, res::CDPSolveResult{VFI},
                 ws::CDPWorkspace, verbose, print_skip)
    operator!(C) = bellman_operator!(cdp, C, ws)
    res.converged, res.num_iter =
        operator_iteration!(operator!, res.C, res.tol, res.max_iter,
                            verbose=verbose, print_skip=print_skip)
    return res
end

# Linear Quadratic Approximation
"""
    LQA

Linear-quadratic approximation algorithm for `solve`.

Use as `solve(cdp, LQA; point=(s, x, e))` to approximate the model around a
reference point and solve the resulting LQ problem.
"""
struct LQA <: DPAlgorithm end

"""
    _solve!(cdp, res, ws, verbose, print_skip; point)

Implement linear quadratic approximation. See `solve` for further details.
"""
function _solve!(cdp::ContinuousDP,
                 res::CDPSolveResult{LQA},
                 ws::CDPWorkspace, verbose, print_skip;
                 point::Tuple{ScalarOrArray, ScalarOrArray, ScalarOrArray})
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
    v_vals = [v(cdp.interp.S[i, :]) for i in 1:length(cdp.interp.basis)]

    # Back out basis coefficients
    ldiv!(res.C, cdp.interp.Phi_lu, v_vals)

    res.converged = true
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
    cdf = cumsum(res.cdp.weights)
    r = rand(rng, ts_length - 1)
    e_ind = Array{Int}(undef, ts_length - 1)
    for t in 1:ts_length - 1
        e_ind[t] = searchsortedlast(cdf, r[t]) + 1
    end

    basis = Basis(map(LinParams, res.eval_nodes_coord, ntuple(i -> 0, N)))
    X_interp = Interpoland(basis, res.X)

    s_ind_front = Base.front(axes(s_path))
    e_ind_tail = Base.tail(axes(res.cdp.shocks))
    view(s_path, (s_ind_front..., 1)... ) .= s_init
    for t in 1:ts_length - 1
        s = s_path[(s_ind_front..., t)...]
        x = X_interp(s)
        e = res.cdp.shocks[(e_ind[t], e_ind_tail...)...]
        view(s_path, (s_ind_front..., t + 1)... ) .= res.cdp.g(s, x, e)
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
