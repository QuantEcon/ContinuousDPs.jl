#=
Tools for solving dynamic programs with continuous states using LQ
approximation.

References
----------
* M. J. Miranda and P. L. Fackler, Applied Computational Economics and Finance,
  MIT press, 2002.

=#
import QuantEcon.LQ, QuantEcon.ScalarOrArray

"""
    approx_lq(s_star, x_star, f_star, Df_star, D²f_star, g_star, Dg_star,
              discount)

Return an approximating LQ instance.

# Arguments
- `s_star::ScalarOrArray{T}`: State variables at the steady-state
- `x_star::ScalarOrArray{T}`: Action variables at the steady-state
- `f_star::Real`: Reward function evaluated at the steady-state
- `Df_star::AbstractVector{T}`: Gradient of f satisfying
  `Df_star = [f_s', f_x']`
- `D²f_star::AbstractMatrix{T}`: Hessian of f satisfying
  `D²f_star::Array = [f_ss f_sx; f_xs f_xx]`
- `g_star::ScalarOrArray{T}`: State transition function evaluated at the
  steady-state
- `Dg_star::AbstractMatrix{T}`: Jacobian of g satisfying `Dg_star = [g_s, g_x]`
- `discount::Real`: Discount factor

"""
function approx_lq(s_star::ScalarOrArray{T}, x_star::ScalarOrArray{T},
                   f_star::Real, Df_star::AbstractVector{T},
                   D²f_star::AbstractMatrix{T}, g_star::ScalarOrArray{T},
                   Dg_star::AbstractMatrix{T}, discount::Real) where T

    n = length(s_star)   # Dim of state variable s
    nb_states = n + 1
    m = length(x_star)  # Dim of control variable x
    z_star = [s_star..., x_star...]

    # Unpack derivatives
    f_s, f_x = Df_star[1:n, :]', Df_star[n+1:end, :]'
    f_ss, f_xs = D²f_star[1:n, 1:n], D²f_star[n+1:end, 1:n]
    f_sx, f_xx = D²f_star[1:n, n+1:end], D²f_star[n+1:end, n+1:end]
    g_s, g_x = Dg_star[:, 1:n], Dg_star[:, n+1:end]

    # Initialize arrays
    A = Array{T}(undef, nb_states, nb_states)
    B = Array{T}(undef, nb_states, m)
    C = zeros(nb_states, 1)
    Q = Array{T}(undef, m, m)
    R = Array{T}(undef, nb_states, nb_states)
    N = Array{T}(undef, m, nb_states)

    # (1, s)' R (1, s) + 2 x' N (1, s) + x' Q x
    R[1, 1] = -(f_star - Df_star' * z_star + z_star' * D²f_star * z_star / 2)
    R[2:end, 1] = -(f_s' - (f_ss * s_star + f_sx * x_star)) / 2
    R[1, 2:end] = -(f_s - (s_star' * f_ss + (f_sx * x_star)')) / 2
    R[2:end, 2:end] = -f_ss / 2

    N[:, 1] = -(f_x' - (f_sx' * s_star + f_xx * x_star)) / 2
    N[:, 2:end] = -f_sx' / 2

    Q[:, :] = -f_xx / 2

    # A (1, s) + B x + C w
    A[1, 1] = 1.0
    A[1, 2:end] .= 0.0
    A[2:end, 1] .= g_star .- Dg_star * z_star  # g_star may be a scalar
    A[2:end, 2:end] = g_s

    B[1, :] .= 0.0
    B[2:end, :] = g_x

    # Construct LQ instance
    lq = QuantEcon.LQ(Q, R, A, B, C, N, bet=discount)

    return lq
end
