using ContinuousDPs: CollocationSolver, LQASolver
using QuantEcon: qnwlogn

@testset "Solver types and primitives-only ContinuousDP" begin
    alpha, beta = 0.4, 0.96
    s_min, s_max = 1e-5, 4.0
    f(s, x) = log(x)
    g(s, x, e) = (s - x)^alpha * e
    x_lb(s) = s_min
    x_ub(s) = s
    shocks, weights = qnwlogn(7, 0.0, 0.1^2)
    basis = Basis(ChebParams(30, s_min, s_max))

    @testset "ContinuousDP constructors" begin
        # Keyword form (canonical)
        cdp = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks,
                           weights=weights, x_lb=x_lb, x_ub=x_ub)
        @test cdp.interp === nothing
        @test cdp.discount == beta
        @test cdp.actions isa ContinuousActions{1}

        # Keyword form with an ActionSpace
        cdp_a = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks,
                             weights=weights,
                             actions=ContinuousActions(x_lb, x_ub))
        @test cdp_a.actions isa ContinuousActions{1}

        # Positional forms
        cdp_p = @inferred ContinuousDP(f, g, beta, shocks, weights, x_lb,
                                       x_ub)
        @test cdp_p.interp === nothing
        cdp_p2 = @inferred ContinuousDP(f, g, beta, shocks, weights,
                                        ContinuousActions(x_lb, x_ub))
        @test cdp_p2.interp === nothing

        # Exclusivity of `actions` and `x_lb`/`x_ub`
        @test_throws ArgumentError ContinuousDP(
            f=f, g=g, discount=beta, shocks=shocks, weights=weights)
        @test_throws ArgumentError ContinuousDP(
            f=f, g=g, discount=beta, shocks=shocks, weights=weights,
            x_lb=x_lb)
        @test_throws ArgumentError ContinuousDP(
            f=f, g=g, discount=beta, shocks=shocks, weights=weights,
            x_lb=x_lb, x_ub=x_ub, actions=ContinuousActions(x_lb, x_ub))
        @test_throws ArgumentError ContinuousDP(
            f=f, g=g, discount=beta, shocks=shocks, weights=weights,
            actions=(x_lb, x_ub))

        # Keyword form with a discrete action space
        x_grid = collect(range(0.1, 0.9, length=9))
        cdp_disc = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks,
                                weights=weights,
                                actions=DiscreteActions(x_grid))
        @test cdp_disc.actions isa DiscreteActions
        @test cdp_disc.actions.vals == x_grid
        @test cdp_disc.interp === nothing

        # The state dimension is not known without a solver
        @test_throws ArgumentError ndims(cdp)
    end

    @testset "CollocationSolver construction" begin
        solver = CollocationSolver(basis)
        @test solver isa CollocationSolver{PFI}
        @test solver.inner_solver == :foc
        @test solver.tol == sqrt(eps())
        @test solver.max_iter == 500

        solver_kw = CollocationSolver(basis=basis, algorithm=VFI,
                                      inner_solver=:brent, tol=1e-6,
                                      max_iter=100)
        @test solver_kw isa CollocationSolver{VFI}
        @test solver_kw.inner_solver == :brent
        @test solver_kw.tol == 1e-6
        @test solver_kw.max_iter == 100

        @test_throws ArgumentError CollocationSolver(basis; algorithm=LQA)
        @test_throws ArgumentError CollocationSolver(basis;
                                                     inner_solver=:newton)

        # Invalid tol / max_iter values are rejected
        @test_throws ArgumentError CollocationSolver(basis; tol=0.0)
        @test_throws ArgumentError CollocationSolver(basis; tol=-1e-8)
        @test_throws ArgumentError CollocationSolver(basis; tol=NaN)
        @test_throws ArgumentError CollocationSolver(basis; tol=Inf)
        # Validation applies to the stored Float64 value: extreme BigFloat
        # inputs that under/overflow in the conversion are rejected
        @test_throws ArgumentError CollocationSolver(basis; tol=big"1e-5000")
        @test_throws ArgumentError CollocationSolver(basis; tol=big"1e5000")

        # max_iter=0 is allowed (fit v_init, iterate zero times); only
        # negative values are rejected
        @test CollocationSolver(basis; max_iter=0).max_iter == 0
        @test_throws ArgumentError CollocationSolver(basis; max_iter=-1)
    end

    @testset "LQASolver construction" begin
        point = (1.0, 0.5, 1.0)
        lqa = LQASolver(basis; point=point)
        @test lqa.point == point
        @test LQASolver(basis=basis, point=point).point == point
    end

    cdp = ContinuousDP(f=f, g=g, discount=beta, shocks=shocks,
                       weights=weights, x_lb=x_lb, x_ub=x_ub)

    @testset "solve with a solver object" begin
        res = @inferred solve(cdp, CollocationSolver(basis); verbose=0)
        @test res.converged
        # The bound problem is stored in the result
        @test res.cdp.interp !== nothing
        @test ndims(res.cdp) == 1
        @test ndims(res) == 1

        # v_init length validation
        @test_throws ArgumentError solve(cdp, CollocationSolver(basis);
                                         v_init=zeros(3), verbose=0)

        # The removed v0.2 method form hits the instructive error stub
        @test_throws ArgumentError solve(cdp, PFI; verbose=0)
        @test_throws ArgumentError solve(cdp; verbose=0)
        @test_throws r"was removed in v0.4" solve(cdp, PFI)
        @test_throws r"was removed in v0.4" solve(cdp, LQA;
                                                  point=(1.0, 0.5, 1.0))

        # A solver is stateless: reuse across solves and problems gives
        # results identical to fresh solves (Interp is built per call)
        solver = CollocationSolver(basis)
        res1 = solve(cdp, solver; verbose=0)
        res2 = solve(cdp, solver; verbose=0)
        @test res2.C == res1.C
        @test res2.V == res1.V
        @test res2.X == res1.X
        res3 = solve(ContinuousDP(cdp; discount=0.9), solver; verbose=0)
        @test res3.converged
        @test res3.C != res1.C
    end

    @testset "removed basis-endowed constructors" begin
        # The v0.2 basis-endowed forms hit the instructive error stubs
        # (the 7-arg form would otherwise silently match the primitives
        # `(..., x_lb, x_ub)` method)
        @test_throws r"removed in v0\.4" ContinuousDP(
            f, g, beta, shocks, weights, x_lb, x_ub, basis)
        @test_throws r"removed in v0\.4" ContinuousDP(
            f, g, beta, shocks, weights, ContinuousActions(x_lb, x_ub),
            basis)

        # The solver's basis determines the approximation
        basis_b = Basis(ChebParams(12, s_min, s_max))
        res_b = solve(cdp, CollocationSolver(basis_b); verbose=0)
        @test res_b.cdp.interp.basis === basis_b
        @test length(res_b.C) == 12
    end

    @testset "copy constructor" begin
        # Primitives copy stays primitives
        cdp2 = @inferred ContinuousDP(cdp)
        @test cdp2.interp === nothing
        @test cdp2.discount == cdp.discount
        @test cdp2.shocks == cdp.shocks
        @test cdp2.weights == cdp.weights

        cdp3 = ContinuousDP(cdp; discount=0.9)
        @test cdp3.discount == 0.9
        @test cdp3.interp === nothing

        # The removed basis keyword is rejected
        @test_throws MethodError ContinuousDP(cdp; basis=basis)

        # A copy of a solver-bound problem is primitives-only
        res = solve(cdp, CollocationSolver(basis); verbose=0)
        cdp4 = ContinuousDP(res.cdp; discount=0.9)
        @test cdp4.interp === nothing
        @test cdp4.discount == 0.9
    end
end
