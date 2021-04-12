using LinearAlgebra
using ContinuousDPs
using QuantEcon: PFI, VFI, solve, stationary_values
using BasisMatrices: Basis, ChebParams, SplineParams, LinParams, nodes
using Test

include("test_cdp.jl")
include("test_lq_approx.jl")
