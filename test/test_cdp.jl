@testset "cdp.jl" begin

    @testset "Deterministic optimal growth" begin
        alpha = 0.65
        beta = 0.95
        f(s, x) = log(x)
        g(s, x, e) = s^alpha - x
        shocks = [1.]
        weights = [1.]
        x_lb(s) = 0
        x_ub(s) = s

        # Analytical solution
        ab = alpha * beta
        c1 = (log(1 - ab) + log(ab) * ab / (1 - ab)) / (1 - beta)
        c2 = alpha / (1 - ab)
        v_star(s) = c1 + c2 * log(s)
        x_star(s) = (1 - ab) * s^alpha

        # Analytical state path
        s_init = 0.1
        ts_length = 25
        s_path_star = Array{Float64}(undef, ts_length)
        s_path_star[1] = s_init
        shock = 1.  # Arbitrary
        for t in 1:ts_length-1
            s_path_star[t+1] = g(s_path_star[t], x_star(s_path_star[t]), shock)
        end

        # Bases
        bases = Basis[]

        # Chebyshev
        n = 30
        s_min, s_max = 0.1, 2.
        push!(bases, Basis(ChebParams(n, s_min, s_max)))

        # Spline
        k = 3
        m = 101
        breaks = m - (k-1)
        s_min, s_max = 0.1, 2.
        push!(bases, Basis(SplineParams(breaks, s_min, s_max, k)))

        methods = [PFI, VFI]

        for basis in bases
            cdp = ContinuousDP(f, g, beta, shocks, weights, x_lb, x_ub, basis)

            for method in methods
                # solve
                tol = sqrt(eps())
                max_iter = 500
                res = @inferred(solve(cdp, method, tol=tol, max_iter=max_iter))

                rtol = 1e-5
                @test isapprox(res.V, v_star.(cdp.interp.S); rtol=rtol)
                @test isapprox(res.X, x_star.(cdp.interp.S); rtol=rtol)

                # set_eval_nodes!
                grid_size = 200
                eval_nodes = collect(range(s_min, stop=s_max, length=grid_size))
                set_eval_nodes!(res, eval_nodes);

                # simulate
                s_path = @inferred(simulate(res, s_init, ts_length))
                atol = 1e-5
                @test isapprox(s_path[end], s_path_star[end]; atol=atol)
            end
        end
    end

end
