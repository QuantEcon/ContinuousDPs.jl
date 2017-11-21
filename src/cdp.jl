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

function Interp(basis::Basis)
    S, Scoord = nodes(basis)
    grid_length = length(basis)
    grid_size = size(basis)
    grid_lb, grid_ub = min(basis), max(basis)
    Phi = BasisMatrix(basis, Expanded(), S).vals[1]
    Phi_lu = lufact(Phi)
    interp = Interp(basis, S, Scoord, grid_length, grid_size, grid_lb, grid_ub,
                    Phi, Phi_lu)
end


mutable struct ContinuousDP{N,TR<:AbstractArray,TS<:VecOrMat,Tf<:Function,Tg<:Function,Tlb<:Function,Tub<:Function}
    f::Tf
    g::Tg
    discount::Float64
    shocks::TR
    weights::Vector{Float64}
    x_lb::Tlb
    x_ub::Tub
    interp::Interp{N,TS}
end

function ContinuousDP(f::Function, g::Function, discount::Float64,
                      shocks::Array{Float64}, weights::Vector{Float64},
                      x_lb::Function, x_ub::Function,
                      basis::Basis)
    interp = Interp(basis)
    cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, interp)
    return cdp
end


mutable struct CDPSolveResult{Algo<:DPAlgorithm,N,TR<:AbstractArray,TS<:VecOrMat}
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

function (res::CDPSolveResult)(s_nodes::AbstractArray{Float64})
    cdp, C = res.cdp, res.C
    V, X = s_wise_max(cdp, s_nodes, C)
    resid = V - vec(funeval(C, cdp.interp.basis, s_nodes))
    return V, X, resid
end


#= Methods =#

function _s_wise_max(cdp::ContinuousDP, s, C)
    sp = Array{Float64}(size(cdp.shocks, 1), length(s))
    function objective(x)
        for i in 1:size(sp, 1)
            sp[i, :] = cdp.g(s, x, cdp.shocks[i, :])
        end
        Vp = funeval(C, cdp.interp.basis, sp)
        cont = cdp.discount * dot(cdp.weights, Vp)
        flow = cdp.f(s, x)
        -1*(flow + cont)
    end
    res = Optim.optimize(objective, cdp.x_lb(s), cdp.x_ub(s))
    v = -res.minimum::Float64
    x = res.minimizer::Float64
    return v, x
end

function s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(indices(ss))
    for i in 1:n
        Tv[i], _ = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return Tv
end

function s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64},
                     X::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(indices(ss))
    for i in 1:n
        Tv[i], X[i] = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return Tv, X
end

function s_wise_max(cdp::ContinuousDP, ss::AbstractArray{Float64},
                    C::Vector{Float64})
    n = size(ss, 1)
    Tv, X = Array{Float64}(n), Array{Float64}(n)
    s_wise_max!(cdp, ss, C, Tv, X)
end


function bellman_operator!(cdp::ContinuousDP, C::Vector{Float64},
                           Tv::Vector{Float64})
    Tv = s_wise_max!(cdp, cdp.interp.S, C, Tv)
    A_ldiv_B!(C, cdp.interp.Phi_lu, Tv)
    return C
end


function compute_greedy!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                         C::Vector{Float64}, X::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(indices(ss))
    for i in 1:n
        _, X[i] = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return X
end

compute_greedy!(cdp::ContinuousDP, C::Vector{Float64}, X::Vector{Float64}) =
    compute_greedy!(cdp, cdp.interp.S, C, X)


function evaluate_policy!(cdp::ContinuousDP{N}, X::Vector{Float64},
                          C::Vector{Float64}) where N
    n = size(cdp.interp.S, 1)
    ts = Base.tail(indices(cdp.interp.S))
    te = Base.tail(indices(cdp.shocks))
    A = Array{Float64}(n, n)
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
    A_lu = lufact(A)
    for i in 1:n
        s = cdp.interp.S[(i, ts...)...]
        C[i] = cdp.f(s, X[i])
    end
    A_ldiv_B!(A_lu, C)
    return C
end


function policy_iteration_operator!(cdp::ContinuousDP, C::Vector{Float64},
                                    X::Vector{Float64})
    compute_greedy!(cdp, C, X)
    evaluate_policy!(cdp, X, C)
    return C
end


function operator_iteration!(T::Function, C::TC, tol::Float64, max_iter;
                             verbose::Int=2, print_skip::Int=50) where TC
    converged = false
    i = 0
    err = tol + 1
    C_old = similar(C)
    while true
        copy!(C_old, C)
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
            warn("max_iter attained")
        elseif verbose == 2
            println("Converged in $i steps")
        end
    end

    return converged, i
end


#= Solve methods =#

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


# Policy iteration
function _solve!(cdp::ContinuousDP, res::CDPSolveResult{PFI},
                 verbose, print_skip)
    C = res.C
    X = Array{Float64}(cdp.interp.length)
    operator!(C) = policy_iteration_operator!(cdp, C, X)
    res.converged, res.num_iter =
        operator_iteration!(operator!, res.C, res.tol, res.max_iter,
                            verbose=verbose, print_skip=print_skip)
    return res
end


# Value iteration
function _solve!(cdp::ContinuousDP, res::CDPSolveResult{VFI},
                 verbose, print_skip)
    C = res.C
    Tv = Array{Float64}(cdp.interp.length)
    operator!(C) = bellman_operator!(cdp, C, Tv)
    res.converged, res.num_iter =
        operator_iteration!(operator!, res.C, res.tol, res.max_iter,
                            verbose=verbose, print_skip=print_skip)
    return res
end


#= Simulate methods =#

function simulate!(rng::AbstractRNG, s_path::TS,
                   res::CDPSolveResult{Algo,N,TR,TS},
                   s_init) where {Algo,N,TR,TS<:VecOrMat}
    ts_length = size(s_path)[end]
    cdf = cumsum(res.cdp.weights)
    r = rand(rng, ts_length-1)
    e_ind = Array{Int}(ts_length-1)
    for t in 1:ts_length-1
        e_ind[t] = searchsortedlast(cdf, r[t]) + 1
    end

    basis = Basis(map(LinParams, res.eval_nodes_coord, ntuple(i -> 0, N)))
    X_interp = Interpoland(basis, res.X)

    s_ind_front = Base.front(indices(s_path))
    e_ind_tail = Base.tail(indices(res.cdp.shocks))
    s_path[(s_ind_front..., 1)...] = s_init
    for t in 1:ts_length-1
        s = s_path[(s_ind_front..., t)...]
        x = X_interp(s)
        e = res.cdp.shocks[(e_ind[t], e_ind_tail...)...]
        s_path[(s_ind_front..., t+1)...] = res.cdp.g(s, x, e)
    end

    return s_path
end

simulate!(s_path::VecOrMat{Float64}, res::CDPSolveResult, s_init) =
    simulate!(Base.GLOBAL_RNG, s_path, res, s_init)


function simulate(rng::AbstractRNG, res::CDPSolveResult{Algo,1}, s_init::Real,
                  ts_length::Integer) where {Algo<:DPAlgorithm}
    s_path = Array{Float64}(ts_length)
    simulate!(rng, s_path, res, s_init)
    return s_path
end

simulate(res::CDPSolveResult{Algo,1}, s_init::Real,
         ts_length::Integer) where {Algo<:DPAlgorithm} =
    simulate(Base.GLOBAL_RNG, res, s_init, ts_length)


function simulate(rng::AbstractRNG, res::CDPSolveResult, s_init::Vector,
                  ts_length::Integer)
    s_path = Array{Float64}(length(s_init), ts_length)
    simulate!(rng, s_path, res, s_init)
    return s_path
end

simulate(res::CDPSolveResult, s_init::Vector, ts_length::Integer) =
    simulate(Base.GLOBAL_RNG, res, s_init, ts_length)
