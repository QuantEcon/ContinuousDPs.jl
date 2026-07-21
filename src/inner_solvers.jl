# Per-state inner maximization and its sweeps over the interpolation
# nodes: the Brent, discrete-enumeration, first-order-condition (FOC),
# and multi-dimensional continuous-action solvers. The operators in
# cdp.jl dispatch into these.

"""
    _s_wise_max!(cdp, ker, s, C, fec)

Find the optimal value and action at a given state `s`.

# Arguments

- `cdp::ContinuousDP`: The dynamic program.
- `ker::_TransitionKernel`: Its transition kernel (see `_build_kernel`).
- `s`: State point at which to maximize.
- `C`: Basis coefficient vector for the value function.
- `fec::FunEvalCache`: Workspace for evaluating the value function at the
  next states.

# Returns

- `v::Float64`: Optimal value at `s`.
- `x::Float64`: Optimal action at `s`.
"""
function _s_wise_max!(cdp::ContinuousDP, ker::_TransitionKernel, s, C,
                      fec::FunEvalCache)
    cdp.actions isa ContinuousActions || throw(ArgumentError(
        "Brent maximization requires a continuous action space; discrete " *
        "action spaces are solved by enumeration"))
    function objective(x)
        cont = _expected_value(ker, fec, C, s, x)
        flow = cdp.f(s, x)
        return flow + cdp.discount * cont
    end
    res = Optim.maximize(objective, cdp.actions.x_lb(s), cdp.actions.x_ub(s))
    v = Optim.maximum(res)::Float64
    x = Optim.maximizer(res)::Float64
    return v, x
end


#= Discrete-action inner solver =#

# Inner objective H(x) = f(s, x) + beta * E[V^(g(s, x, e))]; short-circuits
# on non-finite flow rewards (infeasible actions)
function _objective(cdp::ContinuousDP, ker::_TransitionKernel, s, C,
                    fec::FunEvalCache, x)
    flow = cdp.f(s, x)
    isfinite(flow) || return flow
    cont = _expected_value(ker, fec, C, s, x)
    return flow + cdp.discount * cont
end

"""
    _s_wise_max_discrete!(cdp, ker, s, C, fec)

Find the optimal value and action at state `s` by enumeration over the
discrete action set.

# Returns

- `v::Float64`: Optimal value at `s` (`-Inf` if every action is infeasible).
- `k::Int`: Index of the optimal action in `cdp.actions.vals` (the first
  index if every action is infeasible; ties go to the lowest index).
"""
function _s_wise_max_discrete!(cdp::ContinuousDP, ker::_TransitionKernel,
                               s, C, fec::FunEvalCache)
    vals = cdp.actions.vals
    v, k = -Inf, 1
    for kc in eachindex(vals)
        H = _objective(cdp, ker, s, C, fec, vals[kc])
        if H > v
            v, k = H, kc
        end
    end
    return v, k
end

# Enumeration over all interpolation nodes, storing values in `Tv` and
# action indices in `X_ind`
function _s_wise_max_discrete_sweep!(cp::_CollocationProblem,
                                     C::Vector{Float64},
                                     Tv::Vector{Float64}, X_ind::Vector{Int},
                                     fec::FunEvalCache)
    cdp, ss = cp.cdp, cp.interp.S
    ker = _build_kernel(cp)
    for i in 1:size(ss, 1)
        Tv[i], X_ind[i] = _s_wise_max_discrete!(cdp, ker, _row(ss, i), C,
                                                fec)
    end
    return Tv, X_ind
end


#= First-order-condition inner solver =#

# Relative step for central finite differences of the user's f and g
const _FOC_RELSTEP = cbrt(eps(Float64))

"""
    _objective_and_deriv(cdp, ker, s, C, fec, dfecs, x, x_lb, x_ub)

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
function _objective_and_deriv(cdp::ContinuousDP, ker::_TransitionKernel,
                              s, C, fec::FunEvalCache,
                              dfecs, x::Float64, x_lb::Float64,
                              x_ub::Float64)
    h = min(_FOC_RELSTEP * max(abs(x), 1.0), x - x_lb, x_ub - x)
    h > eps() * max(abs(x), 1.0) || return NaN, NaN
    f0 = cdp.f(s, x)
    fp = (cdp.f(s, x + h) - cdp.f(s, x - h)) / (2h)
    w = _branch_weights(ker, s, x)
    cont = 0.0
    contp = 0.0
    for j in eachindex(w)
        s_next = _branch_state(ker, s, x, j)
        s_up = _branch_state(ker, s, x + h, j)
        s_dn = _branch_state(ker, s, x - h, j)
        v = funeval_point!(fec, C, s_next)
        dv = _grad_dot_gx(dfecs, s_next, s_up, s_dn, h)
        cont += w[j] * v
        contp += w[j] * dv
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
    _s_wise_max_foc!(cdp, ker, s, C, fec, dfecs, x_prev)

Find the optimal value and action at state `s` by solving the first-order
condition `H'(x) = 0` with safeguarded bracketing root-finding (regula falsi
with the Illinois modification), warm-started at `x_prev` (`NaN` for a cold
start). Falls back to the Brent-based `_s_wise_max!` whenever the objective
or its derivative is non-finite at a required point. Corner solutions are
detected from the sign of `H'` during the bracketing expansion.

The coefficients of `dfecs` must have been set with `set_coefs!(., C)`.
"""
function _s_wise_max_foc!(cdp::ContinuousDP, ker::_TransitionKernel, s, C,
                          fec::FunEvalCache, dfecs, x_prev::Float64)
    lb, ub = Float64(cdp.actions.x_lb(s)), Float64(cdp.actions.x_ub(s))
    width = ub - lb
    width > 0 || return _s_wise_max!(cdp, ker, s, C, fec)

    # Evaluation points are kept slightly inside the bounds, where f is more
    # likely to be finite (e.g. log(x) at x_lb = 0)
    off = sqrt(eps()) * width
    lo, hi = lb + off, ub - off

    x0 = isfinite(x_prev) ? clamp(x_prev, lo, hi) : 0.5 * (lo + hi)
    H0, Hp0 = _objective_and_deriv(cdp, ker, s, C, fec, dfecs, x0, lb, ub)
    (isfinite(H0) && isfinite(Hp0)) ||
        return _s_wise_max!(cdp, ker, s, C, fec)

    # Bracket a sign change of H' by expanding from x0 in the uphill
    # direction; `a` keeps H' > 0, `b` keeps H' < 0
    a, Ha, Hpa = x0, H0, Hp0
    b, Hb, Hpb = x0, H0, Hp0
    if Hp0 > 0
        step = 0.02 * width
        while true
            xt = min(a + step, hi)
            Ht, Hpt = _objective_and_deriv(cdp, ker, s, C, fec, dfecs, xt, lb, ub)
            (isfinite(Ht) && isfinite(Hpt)) ||
                return _s_wise_max!(cdp, ker, s, C, fec)
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
            Ht, Hpt = _objective_and_deriv(cdp, ker, s, C, fec, dfecs, xt, lb, ub)
            (isfinite(Ht) && isfinite(Hpt)) ||
                return _s_wise_max!(cdp, ker, s, C, fec)
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
        Hm, Hpm = _objective_and_deriv(cdp, ker, s, C, fec, dfecs, xm, lb, ub)
        (isfinite(Hm) && isfinite(Hpm)) ||
            return _s_wise_max!(cdp, ker, s, C, fec)
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
    _s_wise_max_foc_sweep!(cp, C, Tv, X, fec, dfecs)

Run the FOC-based inner maximization over all interpolation nodes, storing
values in `Tv` and maximizers in `X`. The previous contents of `X` serve as
warm starts (`NaN` entries mean cold start). Sets the coefficients of
`dfecs` from `C`. Falls back to Brent state-by-state on exceptions from the
model functions (e.g. a `DomainError` at a finite-difference point).
"""
function _s_wise_max_foc_sweep!(cp::_CollocationProblem, C::Vector{Float64},
                                Tv::Vector{Float64}, X::Vector{Float64},
                                fec::FunEvalCache, dfecs)
    ker = _build_kernel(cp)
    foreach(dfec -> set_coefs!(dfec, C), dfecs)
    cdp, ss = cp.cdp, cp.interp.S
    for i in 1:size(ss, 1)
        s = _row(ss, i)
        Tv[i], X[i] =
            try
                _s_wise_max_foc!(cdp, ker, s, C, fec, dfecs, X[i])
            catch err
                err isa InterruptException && rethrow()
                _s_wise_max!(cdp, ker, s, C, fec)
            end
    end
    return Tv, X
end

_use_foc(ws::CDPWorkspace) = ws.inner_solver == :foc && ws.dfecs !== nothing


#= Multi-dimensional continuous inner solver =#

# Objective H(x) (and optionally its gradient) for an M-dimensional action
# given as an Optim vector `x`. The gradient combines the exact gradient of
# the fitted value function (via `dfecs`) with bound-aware central finite
# differences of the user's f and g in each action dimension; any non-finite
# intermediate makes the result non-finite, which callers treat as a signal
# to fall back. Writes -H and -grad H (Optim minimizes).
function _negH_multi!(G, cdp::ContinuousDP, ker::_TransitionKernel, s, C,
                      fec::FunEvalCache{N},
                      dfecs, x::Vector{Float64}, lb::Vector{Float64},
                      ub::Vector{Float64}, ::Val{M}) where {N,M}
    # Fminbox inner iterates can momentarily leave the box (the barrier
    # penalizes but does not constrain evaluation points); clamp so that
    # the user's f and g are only ever called at feasible actions
    xt = ntuple(d -> clamp(x[d], lb[d], ub[d]), Val(M))
    flow = cdp.f(s, xt)
    if !isfinite(flow)
        G === nothing || fill!(G, NaN)
        return -flow  # +Inf (or NaN): Optim sees an infeasible point
    end

    if G === nothing
        cont = _expected_value(ker, fec, C, s, xt)
        return -(flow + cdp.discount * cont)
    end

    # Bound-aware finite-difference steps per action dimension (relative
    # to the clamped coordinates)
    h = ntuple(Val(M)) do d
        hd = min(_FOC_RELSTEP * max(abs(xt[d]), 1.0), xt[d] - lb[d],
                 ub[d] - xt[d])
        hd > eps() * max(abs(xt[d]), 1.0) ? hd : NaN
    end
    if !all(isfinite, h)
        fill!(G, NaN)
        return NaN
    end

    # f_x by central differences
    for d in 1:M
        fu = cdp.f(s, ntuple(k -> k == d ? xt[k] + h[d] : xt[k], Val(M)))
        fl = cdp.f(s, ntuple(k -> k == d ? xt[k] - h[d] : xt[k], Val(M)))
        G[d] = (fu - fl) / (2 * h[d])
    end

    w = _branch_weights(ker, s, xt)
    cont = 0.0
    for j in eachindex(w)
        s_next = _branch_state(ker, s, xt, j)
        cont += w[j] * funeval_point!(fec, C, s_next)
        # grad V^(s_next), evaluated once per branch
        gradv = ntuple(nd -> funeval_point!(dfecs[nd], s_next), Val(N))
        for d in 1:M
            s_up = _branch_state(ker, s,
                                 ntuple(k -> k == d ? xt[k] + h[d] : xt[k],
                                        Val(M)), j)
            s_dn = _branch_state(ker, s,
                                 ntuple(k -> k == d ? xt[k] - h[d] : xt[k],
                                        Val(M)), j)
            dv = 0.0
            for nd in 1:N
                dv += gradv[nd] *
                      ((_coord(s_up, nd) - _coord(s_dn, nd)) / (2 * h[d]))
            end
            G[d] += cdp.discount * w[j] * dv
        end
    end

    H = flow + cdp.discount * cont
    for d in 1:M
        G[d] = -G[d]
    end
    return -H
end

"""
    _s_wise_max_multi!(cdp, ker, s, C, fec, dfecs, xout, use_foc)

Find the optimal value and action at state `s` for an `M`-dimensional
continuous action space by box-constrained maximization, warm-started at
`xout` (`NaN` entries mean cold start from the box center, with a coarse
feasible-start probe if that is infeasible). With `use_foc = true` (and
`dfecs` available), `Fminbox(LBFGS)` with the analytic objective gradient
is tried first, falling back to derivative-free cyclic coordinate-wise
Brent maximization on any failure or non-finite outcome; with
`use_foc = false`, coordinate-wise Brent is used directly.

Writes the maximizer into `xout` and returns the maximized value.
"""
function _s_wise_max_multi!(cdp::ContinuousDP, ker::_TransitionKernel,
                            s, C, fec::FunEvalCache,
                            dfecs, xout::AbstractVector{Float64},
                            use_foc::Bool)
    a = cdp.actions
    M = _action_dim(a)
    lb = collect(Float64, a.x_lb(s))
    ub = collect(Float64, a.x_ub(s))
    (length(lb) == M && length(ub) == M) || throw(ArgumentError(
        "x_lb(s) and x_ub(s) must return length-$M bounds"))
    for d in 1:M
        (isfinite(lb[d]) && isfinite(ub[d]) && lb[d] < ub[d]) ||
            throw(ArgumentError(
                "invalid action bounds in dimension $d at state s = $s: " *
                "[$(lb[d]), $(ub[d])]"))
    end
    off = ntuple(d -> sqrt(eps()) * (ub[d] - lb[d]), Val(M))
    x0 = [isfinite(xout[d]) ?
              clamp(xout[d], lb[d] + off[d], ub[d] - off[d]) :
              0.5 * (lb[d] + ub[d]) for d in 1:M]

    # Fminbox requires a finite objective at the starting point; if the
    # warm start (or the box center on a cold start) is infeasible, probe a
    # coarse lattice over the box for the best feasible point
    Hx0 = -_negH_multi!(nothing, cdp, ker, s, C, fec, dfecs, x0, lb, ub, Val(M))
    if !isfinite(Hx0)
        ts = (0.1, 0.3, 0.5, 0.7, 0.9)
        best = -Inf
        cand = similar(x0)
        for I in CartesianIndices(ntuple(_ -> length(ts), Val(M)))
            for d in 1:M
                cand[d] = lb[d] + ts[I[d]] * (ub[d] - lb[d])
            end
            Hc = -_negH_multi!(nothing, cdp, ker, s, C, fec, dfecs, cand, lb,
                               ub, Val(M))
            if isfinite(Hc) && Hc > best
                best = Hc
                copyto!(x0, cand)
            end
        end
        if !isfinite(best)
            # No feasible action found: mirror the discrete-actions
            # convention (value -Inf at the tried point)
            copyto!(xout, x0)
            return -Inf
        end
        Hx0 = best
    end

    if use_foc && dfecs !== nothing
        # Work caps keep near-infeasible corner states bounded (a capped
        # best-effort there is preferable to seconds of barrier refinement)
        opts = Optim.Options(iterations=100, outer_iterations=5)
        # Backtracking line search: only ever shrinks the step, so it stays
        # robust (and cheap) when trial points hit -Inf reward regions
        inner = Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking())
        obj = only_fg!((F, G, x) -> begin
            v = _negH_multi!(G, cdp, ker, s, C, fec, dfecs, x, lb, ub, Val(M))
            F === nothing ? nothing : v
        end)
        r = try
            Optim.optimize(obj, lb, ub, copy(x0), Optim.Fminbox(inner),
                           opts)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        if r !== nothing && isfinite(Optim.minimum(r)) &&
           Optim.minimum(r) <= -Hx0 + sqrt(eps()) * (1 + abs(Hx0))
            copyto!(xout, Optim.minimizer(r))
            return -Optim.minimum(r)
        end
    end

    # Derivative-free path: cyclic coordinate-wise Brent maximization --
    # the M-dimensional generalization of the scalar :brent path, reusing
    # the same robust univariate machinery (Nelder-Mead variants proved
    # unreliable here: Fminbox(NelderMead) can terminate at the starting
    # point, and plain Nelder-Mead stalls against -Inf regions)
    x = copy(x0)
    Hcur = Hx0
    for _ in 1:20
        moved = 0.0
        for d in 1:M
            rd = Optim.optimize(
                t -> begin
                    told = x[d]
                    x[d] = t
                    v = _negH_multi!(nothing, cdp, ker, s, C, fec, dfecs, x,
                                     lb, ub, Val(M))
                    x[d] = told
                    v
                end,
                lb[d], ub[d], Optim.Brent())
            td = Optim.minimizer(rd)
            if isfinite(Optim.minimum(rd)) && -Optim.minimum(rd) >= Hcur
                moved = max(moved, abs(td - x[d]))
                x[d] = td
                Hcur = -Optim.minimum(rd)
            end
        end
        moved <= sqrt(eps()) * max(1.0, maximum(abs, x)) && break
    end
    copyto!(xout, x)
    return Hcur
end

# Sweep over all interpolation nodes for M-dimensional continuous actions;
# rows of `X` are warm starts on input and maximizers on output
function _s_wise_max_multi_sweep!(cp::_CollocationProblem,
                                  C::Vector{Float64},
                                  Tv::Vector{Float64}, X::Matrix{Float64},
                                  fec::FunEvalCache, dfecs, use_foc::Bool)
    if use_foc && dfecs !== nothing
        foreach(dfec -> set_coefs!(dfec, C), dfecs)
    end
    cdp, ss = cp.cdp, cp.interp.S
    ker = _build_kernel(cp)
    for i in 1:size(ss, 1)
        Tv[i] = _s_wise_max_multi!(cdp, ker, _row(ss, i), C, fec, dfecs,
                                   view(X, i, :), use_foc)
    end
    return Tv, X
end
