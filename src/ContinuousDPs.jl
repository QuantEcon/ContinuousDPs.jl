module ContinuousDPs

using QuantEcon
import QuantEcon:
    bellman_operator, bellman_operator!, compute_greedy!, compute_greedy,
    evaluate_policy, DDPAlgorithm, solve, simulate, simulate!

const DPAlgorithm = DDPAlgorithm

include("cdp.jl")

export
    ContinuousDP, evaluate_policy!, set_eval_nodes!, simulate

end # module
