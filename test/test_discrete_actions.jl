using BasisMatrices: funeval
using ContinuousDPs: CDPWorkspace, FunEvalCache, _s_wise_max_discrete!

@testset "DiscreteActions" begin
    s_min, s_max = 0.1, 2.0
    alpha, beta = 0.65, 0.95

    # Deterministic optimal growth with consumption x; infeasible actions
    # (next state outside the domain) get reward -Inf
    f_growth(s, x) =
        (x > 0 && s_min <= s^alpha - x <= s_max) ? log(x) : -Inf
    g_growth(s, x, e) = clamp(s^alpha - x, s_min, s_max)
    shocks, weights = [1.0], [1.0]

    basis = Basis(ChebParams(30, s_min, s_max))
    x_grid = collect(range(1e-3, s_max^alpha, length=501))

    cdp_d = ContinuousDP(f_growth, g_growth, beta, shocks, weights,
                         DiscreteActions(x_grid))
    cdp_c = ContinuousDP(f_growth, g_growth, beta, shocks, weights,
                         s -> 1e-3, s -> s^alpha - s_min)
    # Bound instance for tests of internal functions
    colloc_cdp_d = ContinuousDPs._with_interp(cdp_d, ContinuousDPs.Interp(basis))

    @testset "solve and compare with the continuous solution: $method" for
            method in (PFI, VFI)
        solver = CollocationSolver(basis; algorithm=method)
        res_d = solve(cdp_d, solver, verbose=0)
        res_c = solve(cdp_c, solver, verbose=0)
        @test res_d.converged
        # Action-grid spacing h ~ 3.5e-3: value error is O(h) here (the
        # policy is off by O(h), and V is evaluated along the discrete
        # policy), policy error is O(h)
        h = x_grid[2] - x_grid[1]
        @test maximum(abs, res_d.V .- res_c.V) < 20 * h
        @test maximum(abs, res_d.X .- res_c.X) < 5 * h
        # X and X_ind are consistent, and X values come from the action set
        @test res_d.X == x_grid[res_d.X_ind]
        @test all(in(Set(x_grid)), res_d.X)
    end

    @testset "enumeration agrees with a brute-force reference" begin
        res = solve(cdp_d, CollocationSolver(basis), verbose=0)
        C = res.C
        fec = FunEvalCache(basis)
        for s in (s_min, 0.7, 1.3, s_max)
            v, k = _s_wise_max_discrete!(cdp_d, s, C, fec)
            # reference: objective via funeval at all actions
            H = [begin
                     flow = f_growth(s, x)
                     isfinite(flow) ?
                         flow + beta * funeval(C, basis,
                                               g_growth(s, x, 1.0)) : flow
                 end
                 for x in x_grid]
            @test k == argmax(H)
            @test v ≈ maximum(H) rtol=1e-12
        end
    end

    @testset "workspace and warm-start containers" begin
        ws = CDPWorkspace(colloc_cdp_d)
        @test ws.dfecs === nothing        # no FOC machinery for discrete
        @test length(ws.X_ind) == colloc_cdp_d.interp.length
        @test isempty(ws.X)
    end

    @testset "set_eval_nodes! and callable interface" begin
        res = solve(cdp_d, CollocationSolver(basis), verbose=0)
        grid = collect(range(s_min, s_max, length=100))
        set_eval_nodes!(res, grid)
        @test length(res.X) == length(res.X_ind) == 100
        @test res.X == x_grid[res.X_ind]
        V, X, resid = res(res.eval_nodes)
        @test V == res.V
        @test X == res.X
        @test resid == res.resid
    end

    @testset "simulate uses exact greedy actions" begin
        res = solve(cdp_d, CollocationSolver(basis), verbose=0)
        s_path = simulate(res, 1.0, 30)
        @test all(s_min .<= s_path .<= s_max)
    end

    @testset "non-numeric action values" begin
        # Two-action model with Symbol-valued actions
        moves = Dict(:stay => 0.0, :save => 0.25)
        f2(s, x) = x == :save ? log(0.5 * s) : log(s)
        g2(s, x, e) = clamp(0.5 * s + moves[x], s_min, s_max)
        cdp2 = ContinuousDP(f2, g2, beta, shocks, weights,
                            DiscreteActions([:stay, :save]))
        res2 = solve(cdp2, CollocationSolver(Basis(ChebParams(15, s_min, s_max))), verbose=0)
        @test res2.converged
        @test eltype(res2.X) == Symbol
        @test all(x -> x in (:stay, :save), res2.X)
        @test res2.X == [:stay, :save][res2.X_ind]
    end

    @testset "legacy bellman_operator! overload supports discrete actions" begin
        res = solve(cdp_d, CollocationSolver(basis), verbose=0)
        C0 = copy(res.C)
        n = colloc_cdp_d.interp.length
        ws = CDPWorkspace(colloc_cdp_d)
        Tv = Vector{Float64}(undef, n)
        @test ContinuousDPs.bellman_operator!(colloc_cdp_d, copy(C0), Tv) ==
              ContinuousDPs.bellman_operator!(colloc_cdp_d, copy(C0), ws)
    end

    @testset "continuous action spaces: backward-compatible containers" begin
        # The x_lb/x_ub constructor path must keep the pre-ActionSpace
        # container types (regression tripwire for #88)
        res_c = solve(cdp_c, CollocationSolver(basis), verbose=0)
        @test res_c.X isa Vector{Float64}
        @test isempty(res_c.X_ind)
        @test cdp_c.actions isa ContinuousActions{1}
    end

    @testset "copy constructor and argument errors" begin
        cdp3 = ContinuousDP(cdp_d; discount=0.9)
        @test cdp3.actions === cdp_d.actions
        @test_throws ArgumentError ContinuousDP(cdp_d; x_lb=s -> 0.0)
        @test_throws ArgumentError DiscreteActions(Float64[])
        @test_throws ArgumentError solve(
            cdp_d, LQASolver(basis; point=(1.0, 0.5, 1.0)), verbose=0)
        # inner_solver is validated at solver construction
        @test_throws ArgumentError CollocationSolver(basis;
                                                     inner_solver=:bogus)
        # the action dimension must be a positive integer
        lb1(s) = 0.0
        ub1(s) = 1.0
        @test_throws ArgumentError ContinuousActions{
            0,typeof(lb1),typeof(ub1)}(lb1, ub1)
    end
end
