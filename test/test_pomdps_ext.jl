using ContinuousDPs
using ContinuousDPs: CollocationSolver
using BasisMatrices: Basis, ChebParams, SplineParams
using QuantEcon: qnwlogn
import POMDPs
using POMDPTools: SparseCat, Deterministic, stepthrough, RolloutSimulator
using QuickPOMDPs: QuickMDP
using Random: Xoshiro

const PExt = Base.get_extension(ContinuousDPs, :ContinuousDPsPOMDPsExt)

# A tiny MDP type with a state-dependent action restriction, for the
# feasibility-guard test (QuickMDP's function-form actions does not
# support the zero-arg actions(m) the adapter needs)
struct RestrictedMDP <: POMDPs.MDP{Float64,Symbol} end
POMDPs.actions(::RestrictedMDP) = (:lo, :hi)
POMDPs.actions(::RestrictedMDP, s) = s < 1.0 ? () : (:lo, :hi)
POMDPs.discount(::RestrictedMDP) = 0.9
POMDPs.transition(::RestrictedMDP, s, x) = Deterministic(clamp(s, 0.1, 2.0))
POMDPs.reward(::RestrictedMDP, s, x) = 0.0
POMDPs.isterminal(::RestrictedMDP, s) = false

@testset "POMDPs extension" begin
    alpha, beta = 0.4, 0.95
    s_min, s_max = 0.1, 2.0
    fd(s, x) = log(x)
    gd(s, x, e) = clamp(max(s - x, 1e-8)^alpha * e, s_min, s_max)
    shocks, weights = qnwlogn(3, 0.0, 0.05^2)
    x_grid = collect(range(0.05, 0.5, length=12))
    basis = Basis(ChebParams(20, s_min, s_max))

    cdp_native = ContinuousDP(f=fd, g=gd, discount=beta,
                              actions=DiscreteActions(x_grid),
                              shocks=shocks, weights=weights)
    res_native = solve(cdp_native, CollocationSolver(basis); verbose=0)

    # The same model written as a POMDPs.jl MDP
    m_quick(reward) = QuickMDP(
        statetype = Float64,
        actions = x_grid,
        discount = beta,
        transition = (s, x) -> SparseCat(gd.(s, x, shocks), weights),
        reward = reward,
    )

    @testset "adapter: explicit-finite model, direct reward" begin
        m = m_quick((s, x) -> fd(s, x))
        policy = POMDPs.solve(CollocationSolver(basis), m; verbose=0)
        @test policy.res.converged
        @test policy.res.C ≈ res_native.C rtol=1e-9
        @test policy.res.X == res_native.X
        for s in (0.3, 1.0, 1.7)
            @test POMDPs.action(policy, s) == res_native([s])[2][1]
            @test POMDPs.value(policy, s) ≈ res_native([s])[1][1] rtol=1e-9
        end
        # Full node sweep: the policy is the exact greedy recomputation,
        # and the fitted value differs from V by the residual, which is
        # at solver tolerance at the collocation nodes
        nodes = res_native.eval_nodes
        @test [POMDPs.action(policy, s) for s in nodes] == res_native.X
        @test [POMDPs.value(policy, s) for s in nodes] ≈
              res_native.V - res_native.resid
        @test [POMDPs.value(policy, s) for s in nodes] ≈ res_native.V atol=1e-6
    end

    @testset "adapter: expected-form reward r(s, x, sp)" begin
        # reward defined ONLY in the 4-arg arity, sp-independent, so the
        # expectation reduces to the direct form: identical solution
        m = QuickMDP(
            statetype = Float64,
            actions = x_grid,
            discount = beta,
            transition = (s, x) -> SparseCat(gd.(s, x, shocks), weights),
            reward = (s, x, sp) -> fd(s, x),
        )
        policy = POMDPs.solve(CollocationSolver(basis), m; verbose=0)
        @test policy.res.C ≈ res_native.C rtol=1e-8
    end

    @testset "adapter: simulators run on the policy" begin
        m = m_quick((s, x) -> fd(s, x))
        policy = POMDPs.solve(CollocationSolver(basis), m; verbose=0)
        steps = collect(stepthrough(m, policy, 1.0, "s,a,r", max_steps=10))
        @test length(steps) == 10
        @test all(st -> s_min <= st.s <= s_max, steps)
    end

    @testset "adapter: rollout consistency" begin
        # Cross-validation through the ecosystem: the mean discounted
        # return of seeded rollouts matches the fitted value within
        # Monte Carlo error
        m = m_quick((s, x) -> fd(s, x))
        policy = POMDPs.solve(CollocationSolver(basis), m; verbose=0)
        s0 = 1.0
        rng = Xoshiro(42)
        n_episodes = 200
        returns = [POMDPs.simulate(
                       RolloutSimulator(rng=rng, max_steps=500),
                       m, policy, s0)
                   for _ in 1:n_episodes]
        mc_mean = sum(returns) / n_episodes
        mc_se = sqrt(sum(abs2, returns .- mc_mean) / (n_episodes - 1)) /
                sqrt(n_episodes)
        @test abs(mc_mean - POMDPs.value(policy, s0)) < 5 * mc_se + 0.05
    end

    @testset "adapter: requirement checks" begin
        @test_throws r"no feasible action" POMDPs.solve(
            CollocationSolver(basis), RestrictedMDP(); verbose=0)
        m_term = QuickMDP(
            statetype = Float64,
            actions = x_grid, discount = beta,
            transition = (s, x) -> SparseCat(gd.(s, x, shocks), weights),
            reward = (s, x) -> fd(s, x),
            isterminal = s -> s > 1.9,
        )
        @test_throws r"terminal states are not supported" POMDPs.solve(
            CollocationSolver(basis), m_term; verbose=0)
    end

    @testset "adapter: belief MDP with 2-D state (odu, trimmed)" begin
        # Trimmed version of examples/cdp_ex_odu.jl, built twice: as a
        # native ContinuousDP (callable weights) and as a QuickMDP whose
        # transition carries the same belief-mixed branches
        w_max, c = 2.0, 0.6
        pi_min, pi_max = 1e-3, 1 - 1e-3
        atoms = collect(range(0.05, w_max - 0.05, length=9))
        fdens = fill(1.0 / length(atoms), length(atoms))
        gdens = atoms ./ sum(atoms)
        upd(w, bel) = clamp(
            bel * (1 / length(atoms)) /
            (bel * (1 / length(atoms)) + (1 - bel) * w / sum(atoms)),
            pi_min, pi_max)
        f_odu(s, x) = x === :accept ? s[1] : c
        g_odu(s, x, wp) = x === :accept ? (s[1], s[2]) : (wp, upd(wp, s[2]))
        w_odu(s) = s[2] .* fdens .+ (1 - s[2]) .* gdens
        cdp_odu = ContinuousDP(f=f_odu, g=g_odu, discount=beta,
                               actions=DiscreteActions([:reject, :accept]),
                               shocks=atoms, weights=w_odu)
        basis2 = Basis(
            SplineParams(collect(range(0.0, w_max, length=15)), 0, 3),
            SplineParams(collect(range(pi_min, pi_max, length=15)), 0, 3))
        res_odu = solve(cdp_odu, CollocationSolver(basis2); verbose=0)

        bm = QuickMDP(
            statetype = NTuple{2,Float64},
            actions = [:reject, :accept],
            discount = beta,
            transition = function (s, x)
                x === :accept && return Deterministic((s[1], s[2]))
                SparseCat([(wp, upd(wp, s[2])) for wp in atoms], w_odu(s))
            end,
            reward = (s, x) -> f_odu(s, x),
        )
        policy = POMDPs.solve(CollocationSolver(basis2), bm; verbose=0)
        @test policy.res.converged
        @test policy.res.C ≈ res_odu.C rtol=1e-8
        @test POMDPs.action(policy, (1.8, 0.5)) === :accept
        @test POMDPs.action(policy, (0.2, 0.5)) === :reject
    end

    @testset "model direction (internal as_mdp)" begin
        m = PExt.as_mdp(cdp_native; initialstate=1.0)
        @test m isa POMDPs.MDP{Float64,Float64}
        d = POMDPs.transition(m, 1.0, x_grid[3])
        @test collect(d.vals) == gd.(1.0, x_grid[3], shocks)
        @test collect(d.probs) == weights
        @test POMDPs.reward(m, 1.0, x_grid[3]) == fd(1.0, x_grid[3])
        @test POMDPs.discount(m) == beta
        @test !POMDPs.isterminal(m, 1.0)
        @test POMDPs.actions(m) == x_grid
        @test rand(Xoshiro(0), POMDPs.initialstate(m)) == 1.0
        @test_throws r"no initial state" POMDPs.initialstate(
            PExt.as_mdp(cdp_native))

        # Symbol-valued actions propagate into the action type
        cdp_sym = ContinuousDP(f=(s, x) -> x === :stay ? log(s) : 0.0,
                               g=(s, x, e) -> clamp(s * e, s_min, s_max),
                               discount=beta,
                               actions=DiscreteActions([:stay, :move]),
                               shocks=shocks, weights=weights)
        @test PExt.as_mdp(cdp_sym) isa POMDPs.MDP{Float64,Symbol}

        # Continuous actions: interval action sets, state-dependent only
        cdp_cont = ContinuousDP(f=fd, g=gd, discount=beta,
                                x_lb=s -> 0.01, x_ub=s -> s / 2,
                                shocks=shocks, weights=weights)
        mc = PExt.as_mdp(cdp_cont)
        itv = POMDPs.actions(mc, 1.0)
        @test minimum(itv) == 0.01 && maximum(itv) == 0.5
        @test 0.3 in itv && !(0.6 in itv)
        @test 0.01 <= rand(Xoshiro(0), itv) <= 0.5
        @test_throws r"state-dependent" POMDPs.actions(mc)

        # Round trip: solving through the POMDPs entrance reproduces the
        # native solution exactly (same code path underneath)
        policy = POMDPs.solve(CollocationSolver(basis), mc; verbose=0)
        res_c = solve(cdp_cont, CollocationSolver(basis); verbose=0)
        @test policy.res.C == res_c.C
        # the fitted value differs from the exact-greedy Bellman RHS by
        # the residual at that point
        @test POMDPs.value(policy, 1.2) ≈ res_c([1.2])[1][1] rtol=1e-5

        # 2-D state via statedim: tuple states
        f_2d(s, x) = x === :a ? log(s[1]) : 0.0
        g_2d(s, x, e) = (clamp(s[1] * e, s_min, s_max),
                         clamp(s[2] + 0.1, 0.0, 1.0))
        cdp2 = ContinuousDP(f=f_2d, g=g_2d, discount=beta,
                            actions=DiscreteActions([:a, :b]),
                            shocks=shocks, weights=weights)
        m2 = PExt.as_mdp(cdp2; statedim=2, initialstate=(1.0, 0.5))
        @test m2 isa POMDPs.MDP{NTuple{2,Float64},Symbol}
        d2 = POMDPs.transition(m2, (1.0, 0.5), :a)
        @test all(sp isa NTuple{2,Float64} for sp in d2.vals)
        @test rand(Xoshiro(0), POMDPs.initialstate(m2)) == (1.0, 0.5)
        @test_throws ArgumentError POMDPs.solve(
            CollocationSolver(basis), m2; verbose=0)  # 1-D basis, 2-D state

        # Callable weights are not supported by the model direction
        cdp_cw = ContinuousDP(cdp_native; weights=s -> weights)
        @test_throws r"fixed weights" PExt.as_mdp(cdp_cw)
    end
end
