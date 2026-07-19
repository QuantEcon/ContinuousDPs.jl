using LinearAlgebra
using ContinuousDPs
using QuantEcon: stationary_values, qnwnorm
using BasisMatrices: Basis, ChebParams, SplineParams, LinParams, nodes
using Test
using Random

include("test_point_eval.jl")
include("test_solver_types.jl")
include("test_workspace.jl")
include("test_foc.jl")
include("test_evaluate_policy.jl")
include("test_policy_functions.jl")
include("test_discrete_actions.jl")
include("test_multidim_actions.jl")
include("test_cdp.jl")
include("test_lq_approx.jl")
include("test_cdp_multidim.jl")
