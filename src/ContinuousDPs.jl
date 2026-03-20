module ContinuousDPs

# stdlib
using LinearAlgebra, Random

using QuantEcon
import QuantEcon:
    bellman_operator, bellman_operator!, compute_greedy!, compute_greedy,
    evaluate_policy, DDPAlgorithm, solve, simulate, simulate!

const DPAlgorithm = DDPAlgorithm

include("cdp.jl")
include("lq_approx.jl")

export
    ContinuousDP, evaluate_policy!, set_eval_nodes!, solve, VFI, PFI, LQA,
    simulate, approx_lq

end # module
