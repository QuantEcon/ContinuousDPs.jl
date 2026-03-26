using LinearAlgebra
using ContinuousDPs
using QuantEcon: stationary_values, qnwnorm
using BasisMatrices: Basis, ChebParams, SplineParams, LinParams, nodes
using Test
using Random

include("test_cdp.jl")
include("test_lq_approx.jl")
include("test_cdp_multidim.jl")
