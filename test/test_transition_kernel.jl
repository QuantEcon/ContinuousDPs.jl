using ContinuousDPs: CollocationSolver, CDPWorkspace, FunEvalCache,
                     _colloc, _build_kernel, _forces_brent, _objective,
                     _StateWeights, _StateActionWeights, bellman_operator!,
                     _check_sampling_weights, _TransitionKernel
using BasisMatrices: Basis, ChebParams, funeval
using QuantEcon: qnwlogn
using StaticArrays: SVector

# An RNG replaying a scripted sequence of uniform draws, for testing the
# exact branch-selection conventions of `simulate`
mutable struct ScriptedRNG <: ContinuousDPs.Random.AbstractRNG
    vals::Vector{Float64}
    i::Int
end
ScriptedRNG(vals) = ScriptedRNG(collect(Float64, vals), 0)
ContinuousDPs.Random.rand(rng::ScriptedRNG) = (rng.i += 1; rng.vals[rng.i])
ContinuousDPs.Random.rand(rng::ScriptedRNG, n::Integer) =
    [ContinuousDPs.Random.rand(rng) for _ in 1:n]

# Top-level recorder for the direct _foreach_branch contract test: the
# visited (s_next, weight) pairs accumulate into the explicit mutable
# payload (the contract's no-capturing-closure pattern)
function record_branch(sp, w, states, probs)
    push!(states, sp)
    push!(probs, w)
    return nothing
end

# A minimal kernel implementing only the general _TransitionKernel
# contract (no indexed access): branches materialized as a per-(s, x)
# list, delegating their computation to a wrapped quadrature kernel.
# Stands in for future kernels with (s, x)-dependent supports.
struct ListTestKernel{TK} <: ContinuousDPs._TransitionKernel
    quad::TK
end

function ContinuousDPs._branch_sum(f, ker::ListTestKernel, s, x,
                                   args...)
    q = ker.quad
    w = ContinuousDPs._branch_weights(q, s, x)
    branches = [(ContinuousDPs._branch_state(q, s, x, j), w[j])
                for j in eachindex(w)]
    acc = 0.0
    for (sp, wj) in branches
        acc += f(sp, wj, args...)
    end
    return acc
end

ContinuousDPs._draw_next_state(rng::ContinuousDPs.AbstractRNG,
                               ker::ListTestKernel, s, x) =
    ContinuousDPs._draw_next_state(rng, ker.quad, s, x)

function ContinuousDPs._foreach_branch(f, ker::ListTestKernel, s, x,
                                       args...)
    q = ker.quad
    w = ContinuousDPs._branch_weights(q, s, x)
    for j in eachindex(w)
        f(ContinuousDPs._branch_state(q, s, x, j), w[j], args...)
    end
    return nothing
end

@testset "Callable shock weights" begin
    alpha, beta = 0.4, 0.96
    s_min, s_max = 0.1, 4.0
    f(s, x) = log(x)
    g(s, x, e) = clamp((s - x)^alpha * e, s_min, s_max)
    x_lb(s) = 1e-4
    x_ub(s) = s - 1e-4
    n_shocks = 5
    shocks, weights = qnwlogn(n_shocks, 0.0, 0.05^2)
    basis = Basis(ChebParams(20, s_min, s_max))

    cdp_fixed = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks,
                             weights=weights, x_lb=x_lb, x_ub=x_ub)

    @testset "fixed-weight callables reproduce the fixed path exactly" begin
        w_tuple = Tuple(weights)
        w_svec = SVector{n_shocks}(weights)
        for solver in (CollocationSolver(basis),
                       CollocationSolver(basis; algorithm=VFI, max_iter=80))
            res_f = solve(cdp_fixed, solver; verbose=0)
            for (label, wfun) in [
                    ("vector", s -> weights),
                    ("tuple", s -> w_tuple),
                    ("svector", s -> w_svec),
                    ("tuple(s,x)", (s, x) -> w_tuple),
                ]
                cdp_c = ContinuousDP(cdp_fixed; weights=wfun)
                res_c = solve(cdp_c, solver; verbose=0)
                if label == "tuple(s,x)"
                    # Action-dependent weights force Brent: identical to a
                    # fixed-weight Brent solve instead
                    res_fb = solve(cdp_fixed,
                                   CollocationSolver(
                                       basis;
                                       algorithm=typeof(solver).parameters[1],
                                       inner_solver=:brent,
                                       max_iter=solver.max_iter);
                                   verbose=0)
                    @test res_c.C == res_fb.C
                    @test res_c.X == res_fb.X
                else
                    @test res_c.C == res_f.C
                    @test res_c.V == res_f.V
                    @test res_c.X == res_f.X
                end
            end
        end
    end

    @testset "kernel classification and FOC forcing" begin
        cdp_s = ContinuousDP(cdp_fixed; weights=s -> Tuple(weights))
        cdp_sx = ContinuousDP(cdp_fixed; weights=(s, x) -> Tuple(weights))
        cp_s = _colloc(solve(cdp_s, CollocationSolver(basis); verbose=0))
        cp_sx = _colloc(solve(cdp_sx, CollocationSolver(basis); verbose=0))
        ker_s = _build_kernel(cp_s)
        ker_sx = _build_kernel(cp_sx)
        @test ker_s.weights isa _StateWeights
        @test ker_sx.weights isa _StateActionWeights
        @test !_forces_brent(ker_s)
        @test _forces_brent(ker_sx)
        # State-only weights keep the FOC caches; action-dependent drop them
        @test CDPWorkspace(cp_s).dfecs !== nothing
        @test CDPWorkspace(cp_sx).dfecs === nothing
    end

    # Flavor 2 (issue #110): state-dependent disaster risk p(s)
    p_dis(s) = 0.02 + 0.08 * (s - s_min) / (s_max - s_min)
    e_dis = 0.5
    shocks3 = [e_dis; shocks[2:end]]
    w_dis(s) = (p_dis(s),
                ntuple(j -> (1 - p_dis(s)) * weights[j+1] /
                            sum(weights[2:end]), n_shocks - 1)...)
    cdp_dis = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks3,
                           weights=w_dis, x_lb=x_lb, x_ub=x_ub)

    @testset "flavor 2: disaster risk p(s) end-to-end" begin
        res_pfi = solve(cdp_dis, CollocationSolver(basis); verbose=0)
        res_vfi = solve(cdp_dis, CollocationSolver(basis; algorithm=VFI,
                                                   max_iter=1000);
                        verbose=0)
        @test res_pfi.converged
        @test res_vfi.converged
        @test maximum(abs, res_pfi.C - res_vfi.C) < 1e-5

        # Unit reference: H(s, x) = f + beta * sum_j w_j(s) V(g(s, x, e_j))
        # with V evaluated by BasisMatrices.funeval
        ker = _build_kernel(_colloc(res_pfi))
        fec = FunEvalCache(basis)
        C = res_pfi.C
        for s in (0.5, 1.7, 3.2), x in (0.3 * s, 0.6 * s)
            w = w_dis(s)
            H_ref = f(s, x) + beta * sum(
                w[j] * funeval(C, basis, [g(s, x, shocks3[j])])[1]
                for j in 1:n_shocks)
            @test _objective(cdp_dis, ker, s, C, fec, x) ≈ H_ref rtol=1e-12
        end
    end

    # Flavor 3 (issue #110): action-dependent success probability lambda(x)
    lam(x) = clamp(0.2 + 0.7 * x / s_max, 0.0, 1.0)
    shocks2 = [1.2, 0.8]
    w_lam(s, x) = (lam(x), 1 - lam(x))
    cdp_lam = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks2,
                           weights=w_lam, x_lb=x_lb, x_ub=x_ub)

    @testset "flavor 3: action-dependent weights end-to-end" begin
        res = solve(cdp_lam, CollocationSolver(basis); verbose=0)
        @test res.converged
        ker = _build_kernel(_colloc(res))
        fec = FunEvalCache(basis)
        C = res.C
        for s in (0.5, 1.7, 3.2), x in (0.3 * s, 0.6 * s)
            H_ref = f(s, x) + beta * (
                lam(x) * funeval(C, basis, [g(s, x, shocks2[1])])[1] +
                (1 - lam(x)) * funeval(C, basis, [g(s, x, shocks2[2])])[1])
            @test _objective(cdp_lam, ker, s, C, fec, x) ≈ H_ref rtol=1e-12
        end
        # simulate runs and stays within the domain
        path = simulate(res, 1.0, 100)
        @test all(s -> s_min <= s <= s_max, path)
    end

    @testset "static callable weights keep the sweep allocation profile" begin
        w_tuple = Tuple(weights)
        cdp_c = ContinuousDP(cdp_fixed; weights=s -> w_tuple)
        allocs = map((cdp_fixed, cdp_c)) do cdp
            res = solve(cdp, CollocationSolver(basis); verbose=0)
            cp = _colloc(res)
            ws = CDPWorkspace(cp)
            C = copy(res.C)
            bellman_operator!(cp, C, ws)
            C = copy(res.C)
            @allocated bellman_operator!(cp, C, ws)
        end
        @test allocs[2] == allocs[1]
    end

    @testset "deterministic branch selection in simulate" begin
        # Weights concentrate on one branch depending on the state: the
        # path is deterministic and reproducible by hand
        s_mid = 1.5
        w_det(s) = s < s_mid ? (1.0, 0.0) : (0.0, 1.0)
        cdp_det = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks2,
                               weights=w_det, x_lb=x_lb, x_ub=x_ub)
        res = solve(cdp_det, CollocationSolver(basis); verbose=0)
        path = simulate(res, 1.0, 30)
        pf = ContinuousDPs.PolicyFunction(res)
        s = 1.0
        for t in 2:30
            e = s < s_mid ? shocks2[1] : shocks2[2]
            s = g(s, pf(s), e)
            @test path[t] == s
        end
    end

    @testset "overshooting draws never select a trailing zero branch" begin
        # A draw that lands exactly on cdf[end] (reachable when r * total
        # rounds up) must select the last positive-weight branch, not a
        # trailing zero-probability one. ScriptedRNG feeds r = 1.0 to hit
        # the boundary deterministically; the fixed and callable paths
        # must agree (the callable path's `last` guard vs the fixed
        # path's clamp to the last positive branch).
        shocks3 = [1.2, 0.8, 0.5]
        w3 = [0.5, 0.5, 0.0]
        cdp_zf = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks3,
                              weights=w3, x_lb=x_lb, x_ub=x_ub)
        cdp_zc = ContinuousDP(cdp_zf; weights=s -> (0.5, 0.5, 0.0))
        res_zf = solve(cdp_zf, CollocationSolver(basis); verbose=0)
        res_zc = solve(cdp_zc, CollocationSolver(basis); verbose=0)
        s0 = 1.0
        pf = ContinuousDPs.PolicyFunction(res_zf)
        expected = g(s0, pf(s0), shocks3[2])  # branch 2: last positive
        path_f = simulate(ScriptedRNG([1.0]), res_zf, s0, 2)
        path_c = simulate(ScriptedRNG([1.0]), res_zc, s0, 2)
        @test path_f[2] == expected
        @test path_c[2] == expected
    end

    @testset "general kernel contract (two-tier seam)" begin
        res = solve(cdp_fixed, CollocationSolver(basis); verbose=0)
        cp = _colloc(res)
        quad = _build_kernel(cp)
        gen = ListTestKernel(quad)
        fec = FunEvalCache(basis)
        C = res.C

        # A kernel without indexed access must force Brent
        @test _forces_brent(gen)

        # The general-contract path reproduces the structured kernel
        # through every derivative-free consumer. Agreement is to roundoff,
        # not bitwise: the generic traversal and the specialized loop may
        # compile with different multiply-add fusion (the bit-identity
        # contract applies to the structured path against the pre-kernel
        # code, not between the two tiers).
        for s in (0.5, 1.7, 3.2)
            @test ContinuousDPs._expected_value(gen, fec, C, s, 0.4 * s) ≈
                  ContinuousDPs._expected_value(quad, fec, C, s, 0.4 * s) rtol=1e-14
            v_g, x_g = ContinuousDPs._s_wise_max!(cdp_fixed, gen, s, C, fec)
            v_q, x_q = ContinuousDPs._s_wise_max!(cdp_fixed, quad, s, C,
                                                  fec)
            @test v_g ≈ v_q rtol=1e-12
            @test x_g ≈ x_q atol=1e-6
        end

        # _foreach_branch directly, for both tiers: each branch visited
        # exactly once in branch order, next states and weights forwarded
        # unchanged, the explicit payload reaching the top-level callback,
        # and a `nothing` return
        let s = 1.7, xa = 0.5 * s
            wref = ContinuousDPs._branch_weights(quad, s, xa)
            spref = [ContinuousDPs._branch_state(quad, s, xa, j)
                     for j in eachindex(wref)]
            for ker in (quad, gen)
                states = Float64[]
                probs = Float64[]
                ret = ContinuousDPs._foreach_branch(record_branch, ker, s,
                                                    xa, states, probs)
                @test ret === nothing
                @test states == spref
                @test probs == collect(wref)
            end
        end

        # Discrete enumeration through the general contract
        x_grid = collect(range(0.005, 3.0, length=20))
        fd(s, x) = x <= s - 1e-4 ? log(x) : -Inf
        cdp_d = ContinuousDP(f=fd, g=g, discount=beta, shocks=shocks,
                             weights=weights,
                             actions=DiscreteActions(x_grid))
        res_d = solve(cdp_d, CollocationSolver(basis); verbose=0)
        quad_d = _build_kernel(_colloc(res_d))
        gen_d = ListTestKernel(quad_d)
        for s in (0.5, 1.7, 3.2)
            v_g, k_g = ContinuousDPs._s_wise_max_discrete!(
                cdp_d, gen_d, s, res_d.C, fec)
            v_q, k_q = ContinuousDPs._s_wise_max_discrete!(
                cdp_d, quad_d, s, res_d.C, fec)
            @test v_g ≈ v_q rtol=1e-12
            @test k_g == k_q
        end
    end

    @testset "kernel-carrying problem solves through the general tier" begin
        # A ContinuousDP may carry a ready-made kernel in its weights
        # slot (the injection point used by the generic-model adapter);
        # solving then uses only the general contract: Brent is forced,
        # the policy system assembles through the traversal, simulate
        # draws through _draw_next_state
        res_q = solve(cdp_fixed, CollocationSolver(basis); verbose=0)
        gen = ListTestKernel(_build_kernel(_colloc(res_q)))
        cdp_k = ContinuousDP(f=f, g=nothing, discount=beta,
                             shocks=Float64[], weights=gen,
                             x_lb=x_lb, x_ub=x_ub)
        res_fb = solve(cdp_fixed,
                       CollocationSolver(basis; inner_solver=:brent);
                       verbose=0)
        for alg in (PFI, VFI)
            solver = CollocationSolver(basis; algorithm=alg,
                                       inner_solver=:brent, max_iter=500)
            res_k = solve(cdp_k, solver; verbose=0)
            @test res_k.converged
            res_ref = alg === PFI ? res_fb :
                solve(cdp_fixed, CollocationSolver(basis; algorithm=VFI,
                                                   inner_solver=:brent,
                                                   max_iter=500);
                      verbose=0)
            @test res_k.C ≈ res_ref.C rtol=1e-6
            @test res_k.X ≈ res_ref.X rtol=1e-4
        end
        # FOC request silently degrades to Brent (general kernels force it)
        res_foc = solve(cdp_k, CollocationSolver(basis); verbose=0)
        @test res_foc.C ≈ res_fb.C rtol=1e-6
        # simulate draws through the kernel contract
        path = simulate(ContinuousDPs.Random.Xoshiro(7), res_foc, 1.0, 20)
        @test all(s -> s_min <= s <= s_max, path)
    end

    @testset "validation errors" begin
        # Wrong arity
        cdp_bad = ContinuousDP(cdp_fixed; weights=(a, b, c) -> Tuple(weights))
        @test_throws r"must accept" solve(cdp_bad, CollocationSolver(basis);
                                          verbose=0)
        # Wrong length
        cdp_short = ContinuousDP(cdp_fixed; weights=s -> (0.5, 0.5))
        @test_throws r"one weight per shock node" solve(
            cdp_short, CollocationSolver(basis); verbose=0)
        # Non-collection return
        cdp_scalar = ContinuousDP(cdp_fixed; weights=s -> 1.0)
        @test_throws r"indexable collection" solve(
            cdp_scalar, CollocationSolver(basis); verbose=0)
        # Neither vector nor callable
        @test_throws ArgumentError ContinuousDP(cdp_fixed; weights=1.0)
        # Fixed-weights length validation
        @test_throws r"one weight per shock node" ContinuousDP(
            cdp_fixed; weights=[0.5, 0.5])
    end

    @testset "sampling boundary validation" begin
        # The Bellman operators accept sub-stochastic weights (missing
        # mass = zero continuation value, i.e. extra discounting), but
        # that has no path-wise interpretation: simulate rejects
        # improper weights, for fixed and callable representations alike
        w_sub = [0.4, 0.4]
        cdp_sub = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks2,
                               weights=w_sub, x_lb=x_lb, x_ub=x_ub)
        res_sub = solve(cdp_sub, CollocationSolver(basis); verbose=0)
        @test res_sub.converged   # solving is allowed...
        @test_throws r"summing to one" simulate(res_sub, 1.0, 10)

        cdp_sub_c = ContinuousDP(cdp_sub; weights=s -> (0.4, 0.4))
        res_sub_c = solve(cdp_sub_c, CollocationSolver(basis); verbose=0)
        @test res_sub_c.C == res_sub.C   # same Bellman semantics
        @test_throws r"summing to one" simulate(res_sub_c, 1.0, 10)

        # The boundary check itself: negative, non-finite, and
        # super-stochastic weights are all rejected; proper vectors and
        # quadrature weights pass
        @test_throws r"finite nonnegative" _check_sampling_weights(
            [1.2, -0.2])
        @test_throws r"finite nonnegative" _check_sampling_weights(
            (0.5, NaN))
        @test_throws r"finite nonnegative" _check_sampling_weights(
            [0.5, Inf])
        @test_throws r"summing to one" _check_sampling_weights((0.6, 0.6))
        @test _check_sampling_weights([0.5, 0.5]) == 1.0
        # quadrature weights carry roundoff; the returned total is what
        # the samplers scale their draws by
        @test _check_sampling_weights(Tuple(weights)) ≈ 1 atol=1e-8

        # A proper degenerate distribution: the fixed and callable
        # representations sample the same (here deterministic) process
        cdp_pf = ContinuousDP(cdp_sub; weights=[0.0, 1.0])
        cdp_pc = ContinuousDP(cdp_sub; weights=s -> (0.0, 1.0))
        res_pf = solve(cdp_pf, CollocationSolver(basis); verbose=0)
        res_pc = solve(cdp_pc, CollocationSolver(basis); verbose=0)
        @test simulate(res_pf, 1.0, 20) == simulate(res_pc, 1.0, 20)

        # Branch-selection conventions under scripted draws: the fixed
        # (scaled searchsortedlast) and callable (r < acc) samplers agree
        # at cumulative boundaries, on zero draws with a leading
        # zero-probability branch, and for sums within the accepted
        # tolerance (where draws sample the normalized distribution
        # instead of misassigning the residual mass or indexing out of
        # bounds)
        for (wts, rvals) in [
                ([0.5, 0.5], [0.5, 0.25, 0.75]),
                ([0.0, 1.0], [0.0, 0.5, 1 - 1e-12]),
                ([0.5, 0.5 - 1e-9], [1 - 1e-12, 0.5, 0.0]),
            ]
            cdp_wf = ContinuousDP(cdp_sub; weights=wts)
            wt = Tuple(wts)
            cdp_wc = ContinuousDP(cdp_sub; weights=s -> wt)
            res_wf = solve(cdp_wf, CollocationSolver(basis); verbose=0)
            res_wc = solve(cdp_wc, CollocationSolver(basis); verbose=0)
            pf = simulate(ScriptedRNG(rvals), res_wf, 1.0, length(rvals) + 1)
            pc = simulate(ScriptedRNG(rvals), res_wc, 1.0, length(rvals) + 1)
            @test pf == pc
        end
        # The zero draw on [0.0, 1.0] must select branch 2, not the
        # leading zero-probability branch
        res_z = solve(ContinuousDP(cdp_sub; weights=[0.0, 1.0]),
                      CollocationSolver(basis); verbose=0)
        path_z = simulate(ScriptedRNG([0.0]), res_z, 1.0, 2)
        pfun = ContinuousDPs.PolicyFunction(res_z)
        @test path_z[2] == g(1.0, pfun(1.0), shocks2[2])
    end

    @testset "construction probe tolerates infeasible probe points" begin
        # Discrete actions: a model may define weights only at feasible
        # actions (f == -Inf short-circuits before any weight fetch); the
        # first action being infeasible must not reject the problem at
        # kernel construction
        vals = [-1.0, 0.05, 0.5]
        fd(s, x) = (x < 0 || x >= s) ? -Inf : log(x)
        wd(s, x) = x < 0 ? error("weights undefined at an infeasible action") :
                           (0.5, 0.5)
        cdp_d = ContinuousDP(f=fd, g=g, discount=beta, shocks=shocks2,
                             weights=wd, actions=DiscreteActions(vals))
        res_d = solve(cdp_d, CollocationSolver(basis); verbose=0)
        cdp_df = ContinuousDP(cdp_d; weights=[0.5, 0.5])
        res_df = solve(cdp_df, CollocationSolver(basis); verbose=0)
        @test res_d.C == res_df.C
        @test res_d.X == res_df.X

        # Continuous actions: weights undefined exactly at the lower
        # bound (the probe point); the optimizer only evaluates the
        # interior
        wl(s, x) = x <= x_lb(s) ? error("weights undefined at the bound") :
                                  (0.5, 0.5)
        cdp_l = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks2,
                             weights=wl, x_lb=x_lb, x_ub=x_ub)
        res_l = solve(cdp_l, CollocationSolver(basis); verbose=0)
        cdp_lf = ContinuousDP(cdp_l; weights=[0.5, 0.5])
        res_lf = solve(cdp_lf, CollocationSolver(basis; inner_solver=:brent);
                       verbose=0)
        @test res_l.C == res_lf.C
        @test res_l.X == res_lf.X

        # A tolerated probe failure must not disable validation: a
        # wrong-length return at the first feasible action still errors
        # via the per-fetch length check, rather than silently
        # truncating the branch loop
        w_bad(s, x) = x <= x_lb(s) ? error("undefined at the bound") :
                                     (1.0,)   # one weight, two shocks
        cdp_trunc = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks2,
                                 weights=w_bad, x_lb=x_lb, x_ub=x_ub)
        @test_throws r"one weight per shock node" solve(
            cdp_trunc, CollocationSolver(basis); verbose=0)

        # An actual `nothing` return at the probe point is a malformed
        # return, not a tolerated probe failure
        cdp_nothing = ContinuousDP(cdp_trunc; weights=s -> nothing)
        @test_throws r"indexable collection" solve(
            cdp_nothing, CollocationSolver(basis); verbose=0)
    end
end
