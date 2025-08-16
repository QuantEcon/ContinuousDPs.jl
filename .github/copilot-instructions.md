# ContinuousDPs.jl

ContinuousDPs.jl is a Julia package that provides routines for solving continuous state dynamic programs using the Bellman equation collocation method. It is part of the QuantEcon ecosystem and offers Policy Function Iteration (PFI) and Value Function Iteration (VFI) algorithms for solving dynamic programming problems.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup
- Julia 1.2+ is required. Check version: `julia --version`
- Install package dependencies: `julia --project=. -e "using Pkg; Pkg.instantiate()"` -- takes 75 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
- The package uses standard Julia package structure with `Project.toml` for dependencies

### Building and Testing
- Build the package: `julia --project=. -e "using Pkg; Pkg.build()"`
- Run all tests: `julia --project=. -e "using Pkg; Pkg.test()"` -- takes 60 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
- Run basic functionality test: Create a simple script importing `ContinuousDPs`, `QuantEcon`, and `BasisMatrices` to verify the package works

### Development Workflow
- Start Julia REPL in package mode: `julia --project=.`
- Load the package in development: `julia --project=. -e "using ContinuousDPs"`
- Import required dependencies: `using QuantEcon: PFI, VFI, solve; using BasisMatrices: Basis, ChebParams, SplineParams`
- ALWAYS import `solve` from QuantEcon when using the solving functionality

## Core Functionality Testing

### Creating and Solving a Continuous DP
Always test new code with this basic workflow:
```julia
using ContinuousDPs
using QuantEcon: PFI, VFI, solve
using BasisMatrices: Basis, ChebParams

# Define reward and transition functions
f(s, x) = log(x)  # reward function
g(s, x, e) = s^0.3 * x^0.7 + e  # state transition function

# Setup problem parameters
discount = 0.9
shocks = [0.0]  # shock values
weights = [1.0]  # shock weights
x_lb(s) = 0.01  # lower bound on action
x_ub(s) = s - 0.01  # upper bound on action

# Create basis for approximation
n = 10
s_min, s_max = 0.1, 2.0
basis = Basis(ChebParams(n, s_min, s_max))

# Create continuous DP
cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, basis)

# Solve using PFI or VFI
res = solve(cdp, PFI)  # or VFI

# Simulate solution
s_init = 1.0
ts_length = 10
s_path = simulate(res, s_init, ts_length)
```

### Validation Scenarios
ALWAYS run these validation steps after making changes:
1. **Package Loading Test**: Verify `using ContinuousDPs` works without errors
2. **Basic Solve Test**: Create a simple DP problem and solve it using both PFI and VFI
3. **Simulation Test**: Simulate paths from solved problems to ensure results are reasonable
4. **Integration Test**: Run the full test suite to ensure no regressions

## Repository Structure

### Key Directories and Files
```
├── src/
│   ├── ContinuousDPs.jl     # Main module file
│   ├── cdp.jl               # Core continuous DP functionality
│   └── lq_approx.jl         # Linear quadratic approximation methods
├── test/
│   ├── runtests.jl          # Test runner
│   ├── test_cdp.jl          # Tests for core CDP functionality
│   └── test_lq_approx.jl    # Tests for LQ approximation
├── examples/
│   ├── cdp_ex_optgrowth_jl.ipynb    # Optimal growth model example
│   ├── cdp_ex_MF_jl.ipynb           # Miranda & Fackler examples
│   └── lqapprox_jl.ipynb            # LQ approximation examples
└── Project.toml             # Package dependencies and metadata
```

### Important Files to Check When Making Changes
- Always check `src/cdp.jl` when modifying core solving algorithms
- Always check `src/lq_approx.jl` when working with linear quadratic approximations  
- Always run tests in `test/test_cdp.jl` when modifying CDP functionality
- Always run tests in `test/test_lq_approx.jl` when modifying LQ approximation

## Common Tasks

### Running Examples
- Examples are Jupyter notebooks in the `examples/` directory
- Cannot directly run notebooks in command line, but code can be extracted and run in Julia REPL
- Key examples: optimal growth model, Miranda & Fackler chapter 9 examples

### Algorithm Types
- **PFI (Policy Function Iteration)**: Generally faster convergence
- **VFI (Value Function Iteration)**: More robust but potentially slower
- Both algorithms available through `QuantEcon.solve()` with method parameter

### Basis Types  
- **Chebyshev**: `ChebParams(n, s_min, s_max)` - good general purpose choice
- **Spline**: `SplineParams(breaks, s_min, s_max, k)` - flexible, good for irregular functions
- **Linear**: `LinParams(breaks, s_min, s_max)` - simple linear interpolation

### Debugging Common Issues
- **`solve` not defined**: Import from QuantEcon: `using QuantEcon: solve`
- **Method convergence issues**: Try different basis sizes or algorithm (PFI vs VFI)
- **Simulation errors**: Check that state bounds are consistent with transition function
- **Performance issues**: Larger basis sizes increase accuracy but slow computation

## CI and Quality Assurance

### Continuous Integration
- GitHub Actions runs tests on Ubuntu, Windows, and macOS
- Tests must pass on Julia 1.x (latest stable)
- Nightly CI also runs tests on Julia nightly builds
- CompatHelper automatically updates dependency bounds

### Before Committing Changes
- ALWAYS run `julia --project=. -e "using Pkg; Pkg.test()"` to ensure all tests pass
- Verify basic functionality with a simple CDP example
- Check that examples in `examples/` directory still work if you modified core functionality
- No additional linting or formatting tools required - Julia has built-in code standards

## Dependencies and Compatibility
- **Core dependencies**: QuantEcon.jl, BasisMatrices.jl, Optim.jl, FiniteDiff.jl
- **Julia version**: 1.2+ required (see Project.toml)
- **Platform support**: Windows, macOS, Linux (all tested in CI)
- Dependencies install automatically via `Pkg.instantiate()`

## Performance Notes
- Basis approximation size (`n`) significantly affects both accuracy and speed
- PFI generally converges faster than VFI but requires more memory
- Larger shock grids increase computational complexity
- Simulation is fast once the problem is solved

## Troubleshooting
- **Long solve times**: Normal for complex problems. Increase `max_iter` if needed, but be patient
- **Convergence warnings**: Try different basis size, tolerance, or algorithm
- **Memory issues**: Reduce basis size or use simpler basis types
- **Installation issues**: Ensure Julia 1.2+ and try `Pkg.update()` first