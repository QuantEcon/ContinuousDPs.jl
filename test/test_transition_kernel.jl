using ContinuousDPs: CollocationSolver, CDPWorkspace, FunEvalCache,
                     _colloc, _build_kernel, _forces_brent, _objective,
                     _StateWeights, _StateActionWeights, bellman_operator!
using BasisMatrices: Basis, ChebParams, funeval
using QuantEcon: qnwlogn
using StaticArrays: SVector

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
end
