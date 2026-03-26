# API Reference

```@meta
CurrentModule = ContinuousDPs
```

## Exported

### Model Construction

```@docs
ContinuousDP
ContinuousDP(::Any, ::Any, ::Real, ::AbstractVecOrMat, ::Vector{Float64},
             ::Any, ::Any, ::BasisMatrices.Basis)
ContinuousDP(::ContinuousDP)
```

### Solving the Model

```@docs
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
    "lq_approx.jl",
]
Public = false
Order   = [:type, :function]
```
