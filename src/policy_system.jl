# Assembly and factorization of the policy-evaluation collocation
# system A = Phi - beta * E[Phi(g(S, X, e))], row by row with the
# point-evaluation kernels; dense and sparse paths. Consumed by
# evaluate_policy! in cdp.jl.

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
function _policy_system_lu(Phi::AbstractMatrix, cp::_CollocationProblem,
                           X, fec)
    cdp, ss = cp.cdp, cp.interp.S
    ker = _kernel(cdp)
    n = size(ss, 1)
    A = copyto!(Matrix{Float64}(undef, n, n), Phi)
    for i in 1:n
        s = _row(ss, i)
        x = _row(X, i)
        w = _branch_weights(ker, s, x)
        for j in eachindex(w)
            s_next = _branch_state(ker, s, x, j)
            _sub_basis_row!(A, i, fec, s_next, cdp.discount * w[j])
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
function _policy_system_lu(Phi::SparseMatrixCSC, cp::_CollocationProblem,
                           X, fec)
    cdp, ss = cp.cdp, cp.interp.S
    ker = _kernel(cdp)
    n = size(ss, 1)
    Is, Js, Vs = Int[], Int[], Float64[]
    for i in 1:n
        s = _row(ss, i)
        x = _row(X, i)
        w = _branch_weights(ker, s, x)
        for j in eachindex(w)
            s_next = _branch_state(ker, s, x, j)
            _append_basis_row!(Is, Js, Vs, i, fec, s_next, w[j])
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
