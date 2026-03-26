module ContinuousDPs

# stdlib
using LinearAlgebra, Random

using QuantEcon
import QuantEcon:
    bellman_operator!, compute_greedy!, DDPAlgorithm, VFI, PFI, solve,
    simulate, simulate!

const DPAlgorithm = DDPAlgorithm

include("cdp.jl")
include("lq_approx.jl")

export
    ContinuousDP, solve, VFI, PFI, LQA,
    set_eval_nodes!, simulate, simulate!, approx_lq

end # module
