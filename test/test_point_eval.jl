using BasisMatrices: funeval, evalbase
using ContinuousDPs: FunEvalCache, DerivFunEvalCache, funeval_point!,
                     set_coefs!

@testset "point_eval.jl" begin
    rng = MersenneTwister(0)

    # Evaluation points relative to a domain [a, b]: interior points,
    # breakpoints/endpoints, and points outside the domain
    function eval_points(a, b, breaks=Float64[])
        interior = collect(range(a, stop=b, length=17))[2:2:end]
        vcat(interior, breaks, [a, b, a - 0.1, b + 0.1, a - 2.5, b + 2.5])
    end

    @testset "1d agreement with funeval: $label" for (label, p) in [
        ("Cheb", ChebParams(10, -1.5, 2.5)),
        ("Spline k=1", SplineParams(13, -1.5, 2.5, 1)),
        ("Spline k=2", SplineParams(13, -1.5, 2.5, 2)),
        ("Spline k=3", SplineParams(13, -1.5, 2.5, 3)),
        ("Spline uneven breaks",
         SplineParams([-1.5, -1.2, -0.3, 0., 0.7, 1.9, 2.5], 0, 3)),
        ("Lin", LinParams(11, -1.5, 2.5)),
        ("Lin evennum", LinParams([-1.5, 2.5], 11)),
        ("Lin uneven breaks", LinParams([-1.5, -1.2, -0.3, 0., 0.7, 2.5], 0)),
    ]
        basis = Basis(p)
        C = randn(rng, length(basis))
        fec = FunEvalCache(basis)
        brks = hasproperty(p, :breaks) ? collect(Float64, p.breaks) : Float64[]
        for x in eval_points(min(basis)[1], max(basis)[1], brks)
            expected = funeval(C, basis, x)
            @test funeval_point!(fec, C, x) ≈ expected atol=1e-12 rtol=1e-10
        end
    end

    @testset "Nd agreement with funeval: $label" for (label, basis) in [
        ("Cheb x Cheb",
         Basis(ChebParams(8, 0.1, 2.), ChebParams(5, -1., 1.))),
        ("Spline x Spline",
         Basis(SplineParams(10, 0.1, 2., 3), SplineParams(7, -1., 1., 2))),
        ("Cheb x Spline",
         Basis(ChebParams(8, 0.1, 2.), SplineParams(7, -1., 1., 3))),
        ("Lin x Spline",
         Basis(LinParams(9, 0.1, 2.), SplineParams(7, -1., 1., 3))),
        ("Spline x Cheb x Lin",
         Basis(SplineParams(6, 0.1, 2., 3), ChebParams(4, -1., 1.),
               LinParams(5, 0., 1.))),
    ]
        N = ndims(basis)
        C = randn(rng, length(basis))
        fec = FunEvalCache(basis)
        lb, ub = min(basis), max(basis)
        points = [ntuple(d -> lb[d] + t * (ub[d] - lb[d]), N)
                  for t in [-0.2, 0., 0.13, 0.5, 0.77, 1., 1.2]]
        for x in points
            expected = funeval(C, basis, collect(x))
            # point as a Tuple
            @test funeval_point!(fec, C, x) ≈ expected atol=1e-12 rtol=1e-10
            # point as a Vector
            @test funeval_point!(fec, C, collect(x)) ≈ expected atol=1e-12 rtol=1e-10
        end
    end

    # Reference: expanded derivative basis row built from per-dimension
    # `evalbase` calls. (`funeval` itself cannot serve as the reference for
    # all cases: its `Direct`/`SplineSparse` path errors for `LinParams`
    # derivatives, while the dense `evalbase` supports them.)
    function funeval_deriv(C, basis, x, order)
        N = ndims(basis)
        B = reduce(kron,
                   [Matrix(evalbase(basis.params[d], [Float64(x[d])],
                                    order[d]))
                    for d in N:-1:1])
        return (B * C)[1]
    end

    @testset "derivative agreement with funeval: $label" for (label, basis, orders) in [
        ("Cheb",
         Basis(ChebParams(10, -1.5, 2.5)), [(1,), (2,)]),
        ("Spline k=2",
         Basis(SplineParams(13, -1.5, 2.5, 2)), [(1,)]),
        ("Spline k=3",
         Basis(SplineParams(13, -1.5, 2.5, 3)), [(1,), (2,)]),
        ("Spline uneven breaks",
         Basis(SplineParams([-1.5, -1.2, -0.3, 0., 0.7, 1.9, 2.5], 0, 3)),
         [(1,), (2,)]),
        ("Lin",
         Basis(LinParams(11, -1.5, 2.5)), [(1,)]),
        ("Cheb x Spline",
         Basis(ChebParams(8, 0.1, 2.), SplineParams(7, -1., 1., 3)),
         [(1, 0), (0, 1), (1, 1), (2, 0)]),
        ("Spline x Spline",
         Basis(SplineParams(10, 0.1, 2., 3), SplineParams(7, -1., 1., 2)),
         [(1, 0), (0, 1), (1, 1)]),
        ("Spline x Cheb x Lin",
         Basis(SplineParams(6, 0.1, 2., 3), ChebParams(4, -1., 1.),
               LinParams(5, 0., 1.)),
         [(1, 0, 0), (0, 1, 1)]),
    ]
        N = ndims(basis)
        C = randn(rng, length(basis))
        lb, ub = min(basis), max(basis)
        points = [ntuple(d -> lb[d] + t * (ub[d] - lb[d]), N)
                  for t in [-0.1, 0., 0.13, 0.5, 0.77, 1., 1.1]]
        for order in orders
            dfec = DerivFunEvalCache(basis, order)
            set_coefs!(dfec, C)
            for x in points
                expected = funeval_deriv(C, basis, x, order)
                @test funeval_point!(dfec, N == 1 ? x[1] : x) ≈ expected atol=1e-10 rtol=1e-9
            end
        end
    end

    @testset "derivative cache: set_coefs! updates" begin
        basis = Basis(ChebParams(10, -1.5, 2.5))
        dfec = DerivFunEvalCache(basis, (1,))
        C1, C2 = randn(rng, 10), randn(rng, 10)
        set_coefs!(dfec, C1)
        v1 = funeval_point!(dfec, 0.7)
        set_coefs!(dfec, C2)
        @test funeval_point!(dfec, 0.7) ≈ funeval_deriv(C2, basis, (0.7,), (1,))
        set_coefs!(dfec, C1)
        @test funeval_point!(dfec, 0.7) == v1
    end

    @testset "derivative cache: argument errors" begin
        @test_throws ArgumentError DerivFunEvalCache(
            Basis(SplineParams(13, -1.5, 2.5, 3)), (3,))  # order == degree
        @test_throws ArgumentError DerivFunEvalCache(
            Basis(ChebParams(10, -1.5, 2.5)), (-1,))
    end

    @testset "non-allocating" begin
        alloc_after_warmup(fec, C, x) =
            (funeval_point!(fec, C, x); @allocated funeval_point!(fec, C, x))

        for (basis, x) in [
            (Basis(ChebParams(10, -1.5, 2.5)), 0.7),
            (Basis(SplineParams(13, -1.5, 2.5, 3)), 0.7),
            (Basis(LinParams(11, -1.5, 2.5)), 0.7),
            (Basis(ChebParams(8, 0.1, 2.), SplineParams(7, -1., 1., 3)),
             (0.7, 0.2)),
        ]
            C = randn(rng, length(basis))
            fec = FunEvalCache(basis)
            @test alloc_after_warmup(fec, C, x) == 0
        end

        # derivative evaluation is also non-allocating (after set_coefs!)
        alloc_deriv_after_warmup(dfec, x) =
            (funeval_point!(dfec, x); @allocated funeval_point!(dfec, x))
        basis = Basis(ChebParams(8, 0.1, 2.), SplineParams(7, -1., 1., 3))
        dfec = DerivFunEvalCache(basis, (0, 1))
        set_coefs!(dfec, randn(rng, length(basis)))
        @test alloc_deriv_after_warmup(dfec, (0.7, 0.2)) == 0
    end
end
