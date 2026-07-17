module ContinuousDPs

# stdlib
using LinearAlgebra, Random

using QuantEcon
import QuantEcon:
    bellman_operator!, compute_greedy!, DDPAlgorithm, VFI, PFI, solve,
    simulate, simulate!

const DPAlgorithm = DDPAlgorithm

include("point_eval.jl")
include("cdp.jl")
include("lq_approx.jl")

export
    ContinuousDP, solve, VFI, PFI, LQA,
    CollocationSolver, LQASolver,
    ActionSpace, ContinuousActions, DiscreteActions,
    set_eval_nodes!, simulate, simulate!, approx_lq

end # module
