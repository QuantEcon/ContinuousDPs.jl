using BasisMatrices: ckron, evalbase
using ContinuousDPs: evaluate_policy!, FunEvalCache, Interp, _CollocationProblem
using QuantEcon: qnwnorm

@testset "evaluate_policy!" begin
    # Reference: the pre-optimization implementation (dense, per-point
    # evalbase + ckron row updates)
    function evaluate_policy_ref(colloc_cdp, X)
        cdp, interp = colloc_cdp.cdp, colloc_cdp.interp
        N = ndims(colloc_cdp)
        n = size(interp.S, 1)
        ts = Base.tail(axes(interp.S))
        te = Base.tail(axes(cdp.shocks))
        A = Array{Float64}(undef, n, n)
        A[:] = interp.Phi
        for i in 1:n
            s = interp.S[(i, ts...)...]
            for (j, w) in enumerate(cdp.weights)
                e = cdp.shocks[(j, te...)...]
                s_next = cdp.g(s, X[i], e)
                A[i, :] -= ckron(
                    [vec(evalbase(interp.basis.params[k], s_next[k]))
                     for k in N:-1:1]...
                ) * cdp.discount * w
            end
        end
        C = [cdp.f(interp.S[(i, ts...)...], X[i]) for i in 1:n]
        return A \ C
    end

    shocks, weights = qnwnorm(3, 0.0, 0.01)

    @testset "1d: $label" for (label, basis) in [
        ("Cheb", Basis(ChebParams(15, 0., 1.))),
        ("Spline k=3", Basis(SplineParams(15, 0., 1., 3))),
        ("Lin", Basis(LinParams(15, 0., 1.))),
    ]
        f(s, x) = log(1 + s + x)
        g(s, x, e) = clamp(0.5 * s + 0.2 * x + 0.1 * e, 0., 1.)
        # evaluate_policy! is internal and operates on a problem with a
        # bound interpolation scheme
        colloc_cdp = _CollocationProblem(
            ContinuousDP(f, g, 0.9, shocks, weights, s -> 0., s -> 1.),
            Interp(basis))
        X = [0.25 + 0.5 * s for s in colloc_cdp.interp.S]
        C = Vector{Float64}(undef, colloc_cdp.interp.length)
        evaluate_policy!(colloc_cdp, X, C)
        @test C ≈ evaluate_policy_ref(colloc_cdp, X) rtol=1e-9
        # fec-passing method gives the same result
        C2 = Vector{Float64}(undef, colloc_cdp.interp.length)
        evaluate_policy!(colloc_cdp, X, C2,
                         FunEvalCache(colloc_cdp.interp.basis))
        @test C2 == C
    end

    @testset "2d: $label" for (label, basis) in [
        ("Cheb x Cheb",
         Basis(ChebParams(8, 0., 1.), ChebParams(5, 0., 1.))),
        ("Spline x Spline",
         Basis(SplineParams(8, 0., 1., 3), SplineParams(6, 0., 1., 2))),
        ("Lin x Lin",
         Basis(LinParams(8, 0., 1.), LinParams(6, 0., 1.))),
        ("Cheb x Spline",
         Basis(ChebParams(8, 0., 1.), SplineParams(6, 0., 1., 3))),
    ]
        f(s, x) = log(1 + s[1] + s[2] + x)
        g(s, x, e) = (clamp(0.5 * s[1] + 0.2 * x + 0.1 * e, 0., 1.),
                      clamp(0.8 * s[2] + 0.1, 0., 1.))
        colloc_cdp = _CollocationProblem(
            ContinuousDP(f, g, 0.9, shocks, weights, s -> 0., s -> 1.),
            Interp(basis))
        n = colloc_cdp.interp.length
        X = [0.25 + 0.25 * (colloc_cdp.interp.S[i, 1] +
                            colloc_cdp.interp.S[i, 2])
             for i in 1:n]
        C = Vector{Float64}(undef, n)
        evaluate_policy!(colloc_cdp, X, C)
        @test C ≈ evaluate_policy_ref(colloc_cdp, X) rtol=1e-9
    end
end
