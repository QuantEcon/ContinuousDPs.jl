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
    evaluate!(res)

Evaluate the value function and the policy function at the evaluation nodes.

# Arguments

- `res::CDPSolveResult`: Solution object to update in place.
"""
function evaluate!(res::CDPSolveResult)
    cdp, C, s_nodes = res.cdp, res.C, res.eval_nodes
    res.V, res.X = s_wise_max(cdp, s_nodes, C)
    res.resid = res.V - vec(funeval(C, cdp.interp.basis, s_nodes))
    return res
end

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
    V, X = s_wise_max(cdp, s_nodes, C)
    resid = V - vec(funeval(C, cdp.interp.basis, s_nodes))
    return V, X, resid
end


#= Methods =#

"""
    _s_wise_max!(cdp, s, C, sp)

Find the optimal value and action at a given state `s`.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `s`: State point at which to maximize.
- `C`: Basis coefficient vector for the value function.
- `sp::Matrix{Float64}`: Workspace for next-state evaluations.

# Returns

- `v::Float64`: Optimal value at `s`.
- `x::Float64`: Optimal action at `s`.
"""
function _s_wise_max!(cdp::ContinuousDP, s, C, sp::Matrix{Float64})
    shock_tail = Base.tail(axes(cdp.shocks))

    function objective(x)
        for i in 1:size(sp, 1)
            sp[i, :] .= cdp.g(s, x, cdp.shocks[(i, shock_tail...)...])
        end
        Vp = funeval(C, cdp.interp.basis, sp)
        cont = cdp.discount * dot(cdp.weights, Vp)
        flow = cdp.f(s, x)
        -1 * (flow + cont)
    end
    res = Optim.optimize(objective, cdp.x_lb(s), cdp.x_ub(s))
    v = -res.minimum::Float64
    x = res.minimizer::Float64
    return v, x
end

"""
    s_wise_max!(cdp, ss, C, Tv)

Find optimal value for each grid point.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `Tv::Vector{Float64}`: A buffer array to hold the updated value function.

# Returns

- `Tv::Vector{Float64}`: Updated value function vector.
"""
function s_wise_max!(cdp::ContinuousDP{N}, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64}) where {N}
    n = size(ss, 1)
    t = Base.tail(axes(ss))
    sp = Matrix{Float64}(undef, size(cdp.shocks, 1), N)
    for i in 1:n
        Tv[i], _ = _s_wise_max!(cdp, ss[(i, t...)...], C, sp)
    end
    return Tv
end

"""
    s_wise_max!(cdp, ss, C, Tv, X)

Find optimal value and action for each grid point.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `Tv::Vector{Float64}`: A buffer array to hold the updated value function.
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.

# Returns

- `Tv::Vector{Float64}`: Updated value function vector.
- `X::Vector{Float64}`: Updated policy function vector.
"""
function s_wise_max!(cdp::ContinuousDP{N}, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64},
                     X::Vector{Float64}) where {N}
    n = size(ss, 1)
    t = Base.tail(axes(ss))
    sp = Matrix{Float64}(undef, size(cdp.shocks, 1), N)
    for i in 1:n
        Tv[i], X[i] = _s_wise_max!(cdp, ss[(i, t...)...], C, sp)
    end
    return Tv, X
end

"""
    s_wise_max(cdp, ss, C)

Find optimal value and action for each grid point.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.

# Returns

- `Tv::Vector{Float64}`: Value function vector.
- `X::Vector{Float64}`: Policy function vector.
"""
function s_wise_max(cdp::ContinuousDP, ss::AbstractArray{Float64},
                    C::Vector{Float64})
    n = size(ss, 1)
    Tv, X = Array{Float64}(undef, n), Array{Float64}(undef, n)
    s_wise_max!(cdp, ss, C, Tv, X)
end


"""
    bellman_operator!(cdp, C, Tv)

Apply the Bellman operator and update the basis coefficients. Values are stored
in `Tv`.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `Tv::Vector{Float64}`: Vector to store values.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function bellman_operator!(cdp::ContinuousDP, C::Vector{Float64},
                           Tv::Vector{Float64})
    Tv = s_wise_max!(cdp, cdp.interp.S, C, Tv)
    ldiv!(C, cdp.interp.Phi_lu, Tv)
    return C
end


"""
    compute_greedy!(cdp, C, X)
    compute_greedy!(cdp, ss, C, X)

Compute the greedy policy for the given basis coefficients.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `ss::AbstractArray{Float64}`: Interpolation nodes.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.

# Returns

- `X::Vector{Float64}`: Updated policy function vector.
"""
function compute_greedy!(cdp::ContinuousDP{N}, ss::AbstractArray{Float64},
                         C::Vector{Float64}, X::Vector{Float64}) where {N}
    n = size(ss, 1)
    t = Base.tail(axes(ss))
    sp = Matrix{Float64}(undef, size(cdp.shocks, 1), N)
    for i in 1:n
        _, X[i] = _s_wise_max!(cdp, ss[(i, t...)...], C, sp)
    end
    return X
end

compute_greedy!(cdp::ContinuousDP, C::Vector{Float64}, X::Vector{Float64}) =
    compute_greedy!(cdp, cdp.interp.S, C, X)

"""
    evaluate_policy!(cdp, X, C)

Compute the value function for a given policy and update the basis coefficients.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `X::Vector{Float64}`: Policy function vector.
- `C::Vector{Float64}`: A buffer array to hold the basis coefficients.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function evaluate_policy!(cdp::ContinuousDP{N}, X::Vector{Float64},
                          C::Vector{Float64}) where N
    n = size(cdp.interp.S, 1)
    ts = Base.tail(axes(cdp.interp.S))
    te = Base.tail(axes(cdp.shocks))
    A = Array{Float64}(undef, n, n)
    A[:] = cdp.interp.Phi
    for i in 1:n
        s = cdp.interp.S[(i, ts...)...]
        for (j, w) in enumerate(cdp.weights)
            e = cdp.shocks[(j, te...)...]
            s_next = cdp.g(s, X[i], e)
            A[i, :] -= ckron(
                [vec(evalbase(cdp.interp.basis.params[k], s_next[k]))
                 for k in N:-1:1]...
            ) * cdp.discount * w
        end
    end
    A_lu = lu(A)
    for i in 1:n
        s = cdp.interp.S[(i, ts...)...]
        C[i] = cdp.f(s, X[i])
    end
    ldiv!(A_lu, C)
    return C
end


"""
    policy_iteration_operator!(cdp, C, X)

Perform one step of policy function iteration and update the basis coefficients.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `C::Vector{Float64}`: Basis coefficient vector for the value function.
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.

# Returns

- `C::Vector{Float64}`: Updated basis coefficient vector.
"""
function policy_iteration_operator!(cdp::ContinuousDP, C::Vector{Float64},
                                    X::Vector{Float64})
    compute_greedy!(cdp, C, X)
    evaluate_policy!(cdp, X, C)
    return C
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
        err = maximum(abs, C - C_old)
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
- `method::Type{<:DPAlgorithm}(=PFI)`: Solution method. `VFI` for value
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
               kwargs...) where {Algo<:DPAlgorithm,N}
    tol = Float64(tol)
    res = CDPSolveResult{Algo,N}(cdp, tol, max_iter)
    ldiv!(res.C, cdp.interp.Phi_lu, v_init)
    _solve!(cdp, res, verbose, print_skip; kwargs...)
    evaluate!(res)
    return res
end


# Policy iteration
"""
    _solve!(cdp, res, verbose, print_skip)

Implement policy iteration. See `solve` for further details.
"""
function _solve!(cdp::ContinuousDP, res::CDPSolveResult{PFI},
                 verbose, print_skip)
    C = res.C
    X = Array{Float64}(undef, cdp.interp.length)
    operator!(C) = policy_iteration_operator!(cdp, C, X)
    res.converged, res.num_iter =
        operator_iteration!(operator!, res.C, res.tol, res.max_iter,
                            verbose=verbose, print_skip=print_skip)
    return res
end


# Value iteration
"""
    _solve!(cdp, res, verbose, print_skip)

Implement value iteration. See `solve` for further details.
"""
function _solve!(cdp::ContinuousDP, res::CDPSolveResult{VFI},
                 verbose, print_skip)
    C = res.C
    Tv = Array{Float64}(undef, cdp.interp.length)
    operator!(C) = bellman_operator!(cdp, C, Tv)
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
    _solve!(cdp, res, verbose, print_skip; point)

Implement linear quadratic approximation. See `solve` for further details.
"""
function _solve!(cdp::ContinuousDP,
                 res::CDPSolveResult{LQA},
                 verbose, print_skip;
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
