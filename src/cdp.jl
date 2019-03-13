#=
Tools for representing and solving dynamic programs with continuous states.

Implement the Bellman equation collocation method as described in Mirand and
Fackler (2002), Chapter 9.

References
----------
* M. J. Miranda and P. L. Fackler, Applied Computational Economics and Finance,
  MIT press, 2002.

=#
using BasisMatrices
import Optim


#= Types and contructors =#

"""
    Interp{N,TS,TM,TL}

Type that contains information about interpolation

# Fields

- `basis::Basis{N}`: Object that contains interpolation basis information
- `S::TS<:VecOrMat`: Vector or Matrix that contains interpolation nodes
- `Scoord::NTuple{N,Vector{Float64}}` Tuple that contains transformed
  interpolation nodes
- `length::Int`: Degree of interpolation at tensor grid
- `size::NTuple{N,Int}`: Tuple that contains degree of interpolation at each
  dimension
- `lb::NTuple{N,Float64}`: Lower bound of domain
- `ub::NTuple{N,Float64}`: Upper bound of domain
- `Phi::TM<:AbstractMatrix`: Interpolation basis matrix
- `Phi_lu::TL<:Factorization`: LU factorized interpolation basis matrix
"""
struct Interp{N,TS<:VecOrMat,TM<:AbstractMatrix,TL<:Factorization}
    basis::Basis{N}
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

Constructor for `Interp`

# Arguments

-`basis::Basis`: Object that contains interpolation basis information
"""
function Interp(basis::Basis)
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
    ContinuousDP{N,TR,TS,Tf,Tg,Tlb,Tub}

Type that reperesents a continuous-state dynamic program

# Fields

- `f::Tf<:Function`: Reward function
- `g::Tg<:Function`: State transition function
- `discount::Float64`: Discount factor
- `shocks::TR<:AbstractVecOrMat`: Random variables' nodes
- `weights::Vector{Float64}`: Random variables' weights
- `x_lb::Tlb<:Function`: Lower bound of action variables
- `x_ub::Tub<:Function`: Upper bound of action variables
- `interp::Interp{N,TS<:VecOrMat}`: Object that contains information about
  interpolation
"""
mutable struct ContinuousDP{N,TR<:AbstractVecOrMat,TS<:VecOrMat,
                            Tf<:Function,Tg<:Function,
                            Tlb<:Function,Tub<:Function}
    f::Tf
    g::Tg
    discount::Float64
    shocks::TR
    weights::Vector{Float64}
    x_lb::Tlb
    x_ub::Tub
    interp::Interp{N,TS}
end

"""
    ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, basis)

Constructor for `ContinuousDP`

# Arguments
- `f::Tf<:Function`: Reward function
- `g::Tg<:Function`: State transition function
- `discount::Float64`: Discount factor
- `shocks::TR<:AbstractVecOrMat`: Random variables' nodes
- `weights::Vector{Float64}`: Random variables' weights
- `x_lb::Tlb<:Function`: Lower bound of action variables
- `x_ub::Tub<:Function`: Upper bound of action variables
- `basis::Basis`: Object that contains interpolation basis information
"""
function ContinuousDP(f::Function, g::Function, discount::Float64,
                      shocks::Array{Float64}, weights::Vector{Float64},
                      x_lb::Function, x_ub::Function,
                      basis::Basis)
    interp = Interp(basis)
    cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, interp)
    return cdp
end


"""
    CDPSolveResult{Algo,N,TR,TS}

Type that contains the solution result of continuous-state dynamic programming

# Fields

- `cdp::ContinuousDP{N,TR,TS}`: Object that contains model paramers
- `tol::Float64`: Convergence criteria
- `max_iter::Int`: Maximum number of iteration
- `C::Vector{Float64}`: Basis coefficients vector
- `converged::Bool`: Bool that shows whether model converges
- `num_iter::Int`: Number of iteration until model converges
- `eval_nodes::TS<:VecOrMat`: Evaluation vector or matrix
- `eval_nodes_coord::NTuple{N,Vector{Float64}}`: Tuple that contains evaluation
  transformed grids
- `V::Vector{Float64}`: Computed value function
- `X::Vector{Float64}`: Computed policy function
- `resid::Vector{Float64}`: Residuals of basis coefficients
"""
mutable struct CDPSolveResult{Algo<:DPAlgorithm,N,
                              TR<:AbstractVecOrMat,TS<:VecOrMat}
    cdp::ContinuousDP{N,TR,TS}
    tol::Float64
    max_iter::Int
    C::Vector{Float64}
    converged::Bool
    num_iter::Int
    eval_nodes::TS
    eval_nodes_coord::NTuple{N,Vector{Float64}}
    V::Vector{Float64}
    X::Vector{Float64}
    resid::Vector{Float64}

    function CDPSolveResult{Algo,N,TR,TS}(
            cdp::ContinuousDP{N,TR,TS}, tol::Float64, max_iter::Integer
        ) where {Algo,N,TR,TS}
        C = zeros(cdp.interp.length)
        converged = false
        num_iter = 0
        eval_nodes = cdp.interp.S
        eval_nodes_coord = cdp.interp.Scoord
        V = Float64[]
        X = Float64[]
        resid = Float64[]
        res = new{Algo,N,TR,TS}(cdp, tol, max_iter, C, converged, num_iter,
                                eval_nodes, eval_nodes_coord, V, X, resid)
        return res
    end
end

Base.ndims(::ContinuousDP{N}) where {N} = N
Base.ndims(::CDPSolveResult{Algo,N}) where {Algo,N} = N

"""
    evaluate!(res)

Evaluate the value function and the policy function at each point

# arguments

- `res::CDPSolveResult`: Object to store the result of dynamic programming
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

Set evaluation nodes

# Arguments

- `res::CDPSolveResult`: Object that contains the result of dynamic programming
- `s_nodes_coord::NTuple{N,AbstractVector}`: Evaluation nodes
""" set_eval_nodes!

function (res::CDPSolveResult)(s_nodes::AbstractArray{Float64})
    cdp, C = res.cdp, res.C
    V, X = s_wise_max(cdp, s_nodes, C)
    resid = V - vec(funeval(C, cdp.interp.basis, s_nodes))
    return V, X, resid
end


#= Methods =#

"""
    _s_wise_max(cdp, s, C)

Find optimal value and policy for each grid point

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `s::AbstractArray{Float64}`: Interpolation nodes
- `C::Vector{Float64}`: Basis coefficients vector

# Returns

- `v::Vector{Float64}`: Updated value function vector
- `x::Vector{Float64}`: Updated policy function vector
"""
function _s_wise_max(cdp::ContinuousDP, s, C)
    sp = Array{Float64}(undef, size(cdp.shocks, 1), length(s))
    function objective(x)
        for i in 1:size(sp, 1)
            sp[i, :] .= cdp.g(s, x, cdp.shocks[i, axes(cdp.shocks)[2:end]...])
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

Find optimal value for each grid point

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `ss::AbstractArray{Float64}`: interpolation nodes
- `C::Vector{Float64}`: Basis coefficients vector
- `Tv::Vector{Float64}`: A buffer array to hold the updated value function

# Returns

- `Tv::Vector{Float64}`: Updated value function vector
"""
function s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(axes(ss))
    for i in 1:n
        Tv[i], _ = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return Tv
end

"""
    s_wise_max!(cdp, ss, C, Tv)

Find optimal value and policy for each grid point

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `ss::AbstractArray{Float64}`: interpolation nodes
- `C::Vector{Float64}`: Basis coefficients vector
- `Tv::Vector{Float64}`: A buffer array to hold the updated value function
- `X::Vector{Float64}`: A buffer array to hold the updeted policy function

# Returns

- `Tv::Vector{Float64}`: Updated value function vector
- `X::Vector{Float64}`: Updated policy function vector
"""
function s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64},
                     X::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(axes(ss))
    for i in 1:n
        Tv[i], X[i] = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return Tv, X
end

"""
    s_wise_max(cdp, ss, C)

Find optimal value and policy for each grid point

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `ss::AbstractArray{Float64}`: Interpolation nodes
- `C::Vector{Float64}`: Basis coefficients vector

# Returns

- `Tv::Vector{Float64}`: Value function vector
- `X::Vector{Float64}`: Policy function vector
"""
function s_wise_max(cdp::ContinuousDP, ss::AbstractArray{Float64},
                    C::Vector{Float64})
    n = size(ss, 1)
    Tv, X = Array{Float64}(undef, n), Array{Float64}(undef, n)
    s_wise_max!(cdp, ss, C, Tv, X)
end


"""
    bellman_operator!(cdp, C, Tv)

Update basis coefficients. Values are stored in `Tv`

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `C::Vector{Float64}`: Basis coefficients vector
- `Tv::Vector{Float64}`: Vector to store values

# Returns

- `C::Vector{Float64}`: Updated basis coefficients vector
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

Updates policy function vector

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `ss::AbstractArray{Float64}`: Interpolation nodes
- `C::Vector{Float64}`: Basis coefficients vector
- `X::Vector{Float64}`: A buffer array to hold the updated policy function.

# Returns

- `X::Vector{Float64}`: Updated policy function vector
"""
function compute_greedy!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                         C::Vector{Float64}, X::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(axes(ss))
    for i in 1:n
        _, X[i] = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return X
end

compute_greedy!(cdp::ContinuousDP, C::Vector{Float64}, X::Vector{Float64}) =
    compute_greedy!(cdp, cdp.interp.S, C, X)

"""
    evaluate_policy!(cdp, X, C)

Update basis coefficients

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `X::Vector{Float64}`: Policy function vector
- `C::Vector{Float64}`: A buffer array to hold the basis coefficients

# Returns

- `C::Vector{Float64}`: Updated basis coefficients vector
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

Update basis coefficients by policy function iteration

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `C::Vector{Float64}`: Basis coefficients vector
- `X::Vector{Float64}`: A buffer array to hold the updated policy function

# Returns

- `C::Vector{Float64}` Updated basis coefficients vector
"""
function policy_iteration_operator!(cdp::ContinuousDP, C::Vector{Float64},
                                    X::Vector{Float64})
    compute_greedy!(cdp, C, X)
    evaluate_policy!(cdp, X, C)
    return C
end


"""
    operator_iteration!(T, C, tol, max_iter; verbose=2, print_skip=50)

Updates basis coefficients until it converges.

# Arguments

- `T::Function`: Function that updates basis coefficients by VFI or PFI
- `C::Vector{Float64}`: initial basis coefficients vector
- `tol::Float64`: Tolerance to be used to update basis coefficients
- `max_iter::Int`: The maximum number of iteration
- `verbose::Int`: Level of feedback (0 for no output, 1 for warnings only, 2 for
   warning and convergence messages during iteration)
- `print_skip::Int`: if verbose == 2, how many iterations to apply between print
  messages

# Returns

- `converged::Bool`: Bool that shows whether basis coefficients vector converges
- `i::Int`: Number of iteration it took to converge
"""
function operator_iteration!(T::Function, C::TC, tol::Float64, max_iter;
                             verbose::Int=2, print_skip::Int=50) where TC
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
    solve(cdp, method=PFI; tol=sqrt(eps()), max_iter=500, verbose=2,
          print_skip=50)
    solve(cdp, method=PFI; v_init, tol=sqrt(eps()), max_iter=500, verbose=2,
          print_skip=50)

Solve the continuous-state dynamic program

# Arguments

- `cdp::ContinuousDP`: Object that contains model parameters
- `method::Type{T<Algo}(PFI)`: Type name specifying solution method
   Acceptable arguments are 'VFI' for value function iteration or
   'PFI' for policy function iteration. Default solution method is 'PFI'.
- `v_init::Vector{Float64}`: Initial guess for value function
- `tol::Real`: Value for epsilon-optimality
- `max_iter::Int`: Maximum number of iterations
- `verbose::Int`: Level of feedback (0 for no output, 1 for warnings only, 2 for
   warning and convergence messages during iteration)
- `print_skip::Int`: if verbose == 2, how many iterations to apply between print
  messages

# Returns

- `res::CDPSolveResult{Algo,N,TR,TS}`: Object to store the result of dynamic
  programming
"""
function solve(cdp::ContinuousDP{N,TR,TS}, method::Type{Algo}=PFI;
               tol::Real=sqrt(eps()), max_iter::Integer=500,
               verbose::Int=2,
               print_skip::Int=50) where {Algo<:DPAlgorithm,N,TR,TS}
    tol = Float64(tol)
    res = CDPSolveResult{Algo,N,TR,TS}(cdp, tol, max_iter)
    _solve!(cdp, res, verbose, print_skip)
    evaluate!(res)
    return res
end

function solve(cdp::ContinuousDP{N,TR,TS}, method::Type{Algo}=PFI;
               v_init::Vector{Float64}
               tol::Real=sqrt(eps()), max_iter::Integer=500,
               verbose::Int=2,
               print_skip::Int=50) where {Algo<:DPAlgorithm,N,TR,TS}
    tol = Float64(tol)
    res = CDPSolveResult{Algo,N,TR,TS}(cdp, tol, max_iter)
    ldiv!(res.C, cdp.interp.Phi_lu, Tv)
    _solve!(cdp, res, verbose, print_skip)
    evaluate!(res)
    return res
end


# Policy iteration
"""
    _solve!(cdp, res, verbose, print_skip)

Implement Policy Iteration. See `solve` for further details.
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

Implement Value Iteration. See `solve` for further details
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


#= Simulate methods =#

"""
    simulate!([rng=GLOBAL_RNG], s_path, res, s_init)

Generate a sample path of state variable(s)

# Arguments

- `rng::AbstractRNG`: Random number generator
- `s_path::VecOrMat`: Array to store the generated sample path
- `res::CDPSolveResult`: Object that contains result of dynamic programming
- `s_init`: Initial value of state variable(s)

# Return

- `s_path::VecOrMat`:: Generated sample path of state variable(s)
"""
function simulate!(rng::AbstractRNG, s_path::TS,
                   res::CDPSolveResult{Algo,N,TR,TS},
                   s_init) where {Algo,N,TR,TS<:VecOrMat}
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
    s_path[(s_ind_front..., 1)...] = s_init
    for t in 1:ts_length - 1
        s = s_path[(s_ind_front..., t)...]
        x = X_interp(s)
        e = res.cdp.shocks[(e_ind[t], e_ind_tail...)...]
        s_path[(s_ind_front..., t + 1)...] = res.cdp.g(s, x, e)
    end

    return s_path
end

simulate!(s_path::VecOrMat{Float64}, res::CDPSolveResult, s_init) =
    simulate!(Random.GLOBAL_RNG, s_path, res, s_init)

"""
    simulate([rng=GLOBAL_RNG], res, s_init, ts_length)

Generate a sample path of state variable(s)

# Arguments

- `rng::AbstractRNG`: Random number generator
- `res::CDPSolveResult`: Object that contains result of dynamic programming
- `s_init`: Initial value of state variable(s)
- `ts_length::Integer`: Length of simulation

# Return

- `s_path::VecOrMat`:: Generated sample path of state variable(s)
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
