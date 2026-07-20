# API Reference

```@meta
CurrentModule = ContinuousDPs
```

## Exported

### Model Construction

```@docs
ContinuousDP
ContinuousDP(::Any, ::Any, ::Real, ::AbstractVecOrMat, ::Vector{Float64},
             ::ActionSpace)
ContinuousDP(::ContinuousDP)
```

### Action Spaces

```@docs
ActionSpace
ContinuousActions
DiscreteActions
```

### Solving the Model

```@docs
CollocationSolver
LQASolver
solve
VFI
PFI
LQA
```

### Evaluation and Simulation

```@docs
set_eval_nodes!
simulate
simulate!
```

### LQ Approximation

```@docs
approx_lq
```

## Internal

```@autodocs
Modules = [ContinuousDPs]
Pages   = [
    "cdp.jl",
    "inner_solvers.jl",
    "policy_system.jl",
    "lq_approx.jl",
]
Public = false
Order   = [:type, :function]
```
