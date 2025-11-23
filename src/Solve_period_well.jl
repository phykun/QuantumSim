export u_nk_Position, get_bond, Blo_Wan_FBZ

function u_nk_Position(state, period_k, x)
    order = (size(state, 1) - 1) / 2
    order_set = collect(-order:order)
    exp_factor = exp.(im * period_k * x * order_set')
    u_nk = exp_factor * state
    for i in axes(u_nk, 2)
        tmp = norm(u_nk[:, i])
        u_nk[:, i] /= tmp
    end
    return u_nk
end

function get_bond(func_H, k, m=1, ħ=1, order::Int=5)
    N = length(k)
    period_k = k[end] - k[1]
    band = zeros(N, 2 * order + 1)
    for i in axes(k, 1)
        Ham = func_H(k[i], period_k, m, ħ)
        D, P = eigen(Ham)
        band[i, :] = D
    end
    return band
end

function Blo_Wan_FBZ(func_H, k, x, Nbond=1, m=1, ħ=1, order::Int=5)
    period_k = k[end] - k[1]
    k_FBZ = k[2:end]
    N_cell = length(k_FBZ)
    Nx = length(x)
    period_x = 2pi / period_k
    state = zeros(2 * order + 1, Nbond * N_cell)
    for i in axes(k_FBZ, 1)
        Ham = func_H(k_FBZ[i], period_k, m, ħ, order)
        D, P = eigen(Ham)
        for j in 1:Nbond
            state[:, i+(j-1)*N_cell] = P[:, j]
        end
    end
    u_k = u_nk_Position(state, period_k, x)
    exp_factor = exp.(im * x * k_FBZ')
    site = range(period_x, x[end], N_cell) .- period_x / 2
    Bloch = zeros(ComplexF64, Nx, Nbond * N_cell)
    Wannier = zeros(ComplexF64, Nx, Nbond * N_cell)
    for j in 1:Nbond
        n1 = (j - 1) * N_cell + 1
        n2 = j * N_cell
        Bloch[:, n1:n2] = exp_factor .* u_k[:, n1:n2]
        Wannier[:, n1:n2] = Bloch[:, n1:n2] * exp.(-im * k_FBZ * site') / sqrt(N_cell)
    end
    return Bloch, Wannier
end