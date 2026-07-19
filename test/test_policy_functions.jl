using ContinuousDPs: ValueFunction, PolicyFunction
using BasisMatrices: Basis, ChebParams, LinParams, SplineParams, Interpoland,
    funeval
using QuantEcon: qnwlogn

@testset "ValueFunction and PolicyFunction" begin
    # 1-D stochastic optimal growth (README model)
    alpha, beta = 0.4, 0.96
    s_min, s_max = 1e-5, 4.0
    f(s, x) = log(x)
    g(s, x, e) = (s - x)^alpha * e
    shocks, weights = qnwlogn(5, 0.0, 0.1^2)
    cdp = ContinuousDP(f=f, g=g, discount=beta, x_lb=s -> s_min, x_ub=s -> s,
                       shocks=shocks, weights=weights)
    basis = Basis(ChebParams(20, s_min, s_max))
    res = solve(cdp, CollocationSolver(basis); verbose=0)

    grid = collect(range(s_min, stop=s_max, length=101))

    @testset "ValueFunction: fitted-V evaluation" begin
        vf = ValueFunction(res)
        @test [vf(s) for s in grid] ≈ vec(funeval(res.C, basis, grid))
        # V - resid at the evaluation nodes equals the fitted V
        @test [vf(s) for s in res.eval_nodes] ≈ res.V - res.resid
    end

    @testset "PolicyFunction: continuous scalar" begin
        pf = PolicyFunction(res)
        # Agreement with the Interpoland-based interpolation (the
        # pre-refactor implementation used by simulate!), up to the clamp:
        # outside the hull of the evaluation nodes the linear interpolant
        # extrapolates, where the unclamped value can violate the action
        # bounds (x > x_ub(s) = s here, making g infeasible)
        lin_basis = Basis(LinParams(res.eval_nodes_coord[1], 0))
        itp = Interpoland(lin_basis, res.X)
        @test [pf(s) for s in grid] ≈ [clamp(itp(s), s_min, s) for s in grid]
        # Exact at the evaluation nodes
        @test [pf(s) for s in res.eval_nodes] ≈ res.X
        # Construction after set_eval_nodes! picks up the new nodes
        set_eval_nodes!(res, collect(range(s_min, stop=s_max, length=57)))
        pf2 = PolicyFunction(res)
        @test [pf2(s) for s in res.eval_nodes] ≈ res.X
    end

    @testset "PolicyFunction: clamping into the action bounds" begin
        # Convex upper bound: linear interpolation of bound-attaining nodal
        # values overshoots the bound mid-interval, so the clamp must bind.
        # The transition function rejects infeasible actions outright, so
        # the simulation below fails unless the clamped evaluator is used.
        x_ub_c(s) = s^2
        fc(s, x) = -(x - 0.5 * s)^2
        function gc(s, x, e)
            x <= x_ub_c(s) || throw(DomainError(x, "infeasible action"))
            return clamp(0.5 * s, 0.5, 2.0)
        end
        cdp_c = ContinuousDP(f=fc, g=gc, discount=0.9, x_lb=s -> 0.0,
                             x_ub=x_ub_c, shocks=[0.0], weights=[1.0])
        res_c = solve(cdp_c, CollocationSolver(Basis(ChebParams(10, 0.5, 2.0)));
                      verbose=0)
        res_c.X .= x_ub_c.(res_c.eval_nodes)   # policy at the (convex) bound
        nodes = res_c.eval_nodes
        s_mid = (nodes[1] + nodes[2]) / 2
        # the raw interpolant is infeasible at s_mid ...
        lin_c = Basis(LinParams(res_c.eval_nodes_coord[1], 0))
        raw = Interpoland(lin_c, res_c.X)
        @test raw(s_mid) > x_ub_c(s_mid)
        # ... PolicyFunction restores feasibility ...
        pf_c = PolicyFunction(res_c)
        @test pf_c(s_mid) == x_ub_c(s_mid)
        # ... and simulate traverses the overshoot region without tripping
        # the transition's feasibility check (regression: fails if simulate!
        # returns to the unclamped interpolation path)
        s_path = simulate(MersenneTwister(7), res_c, s_mid, 5)
        @test all(isfinite, s_path)
    end

    @testset "PolicyFunction: 2-D state (tensor piecewise linear)" begin
        f3(s, x) = log(x)
        g3(s, x, e) = (clamp(s[1] - x + 0.6, 0.5, 2.0),
                       clamp(0.5 * s[2] + 0.5, 0.5, 2.0))
        cdp3 = ContinuousDP(f=f3, g=g3, discount=0.9, x_lb=s -> 0.01,
                            x_ub=s -> s[1], shocks=[0.0], weights=[1.0])
        basis3 = Basis(SplineParams(6, 0.5, 2.0, 2), SplineParams(4, 0.5, 2.0, 2))
        res3 = solve(cdp3, CollocationSolver(basis3; algorithm=VFI,
                                             max_iter=5); verbose=0)
        pf3 = PolicyFunction(res3)
        lin3 = Basis(map(LinParams, res3.eval_nodes_coord, (0, 0))...)
        itp3 = Interpoland(lin3, res3.X)
        for s1 in range(0.5, stop=2.0, length=7), s2 in (0.6, 1.3, 1.9)
            s = [s1, s2]
            @test pf3(s) ≈ clamp(itp3(s), 0.01, s1)
        end
    end

    @testset "PolicyFunction: M-dimensional continuous actions" begin
        f2(s, x) = -(x[1] - 0.5 * s)^2 - (x[2] - 0.25 * s)^2 + log(1 + s)
        g2(s, x, e) = clamp(0.5 * s + 0.1 * x[1], 0.5, 2.0)
        actions2 = ContinuousActions{2}(s -> (0.0, 0.0), s -> (s, s))
        cdp2 = ContinuousDP(f=f2, g=g2, discount=0.9, actions=actions2,
                            shocks=[0.0], weights=[1.0])
        basis2 = Basis(SplineParams(15, 0.5, 2.0, 3))
        res2 = solve(cdp2, CollocationSolver(basis2; algorithm=VFI,
                                             max_iter=10); verbose=0)
        pf2 = PolicyFunction(res2)
        lin_basis2 = Basis(LinParams(res2.eval_nodes_coord[1], 0))
        itps = ntuple(d -> Interpoland(lin_basis2, res2.X[:, d]), 2)
        for s in range(0.5, stop=2.0, length=21)
            x = pf2(s)
            @test x isa NTuple{2,Float64}
            @test collect(x) ≈ [itps[1](s), itps[2](s)]
        end
        @test @inferred(pf2(1.0)) isa NTuple{2,Float64}
    end

    @testset "PolicyFunction: discrete actions (exact greedy)" begin
        fd(s, x) = (x > 0 && 0.1 <= s^0.65 - x <= 2.0) ? log(x) : -Inf
        gd(s, x, e) = clamp(s^0.65 - x, 0.1, 2.0)
        x_grid = collect(range(1e-3, 2.0^0.65, length=101))
        cdp_d = ContinuousDP(f=fd, g=gd, discount=0.95,
                             actions=DiscreteActions(x_grid),
                             shocks=[0.0], weights=[1.0])
        basis_d = Basis(ChebParams(15, 0.1, 2.0))
        res_d = solve(cdp_d, CollocationSolver(basis_d); verbose=0)
        pf_d = PolicyFunction(res_d)
        # Exact greedy at arbitrary points: agrees with the result callable
        ss = collect(range(0.1, stop=2.0, length=17))
        V, X, resid = res_d(ss)
        @test [pf_d(s) for s in ss] == X
        @test all(x -> x in x_grid, pf_d(s) for s in ss)
        @test @inferred(pf_d(1.0)) isa Float64
    end

    @testset "inference and machinery allocations" begin
        # Non-allocating model functions isolate the evaluator machinery
        vf = ValueFunction(res)
        pf = PolicyFunction(res)
        @test @inferred(vf(1.0)) isa Float64
        @test @inferred(pf(1.0)) isa Float64
        vf(1.0); pf(1.0)
        @test iszero(@allocated vf(1.0))
        @test iszero(@allocated pf(1.0))
    end

    @testset "simulate uses the functor policy (regression)" begin
        using Random
        s_path1 = simulate(MersenneTwister(7), res, 1.0, 30)
        s_path2 = simulate(MersenneTwister(7), res, 1.0, 30)
        @test s_path1 == s_path2
        @test all(s_min .<= s_path1 .<= s_max)
    end
end
