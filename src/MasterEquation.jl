export MasterEq, MasterEq2, MasterEq_FSL
# 哈密顿含时间的情况还没经过实例检验
# 如果H是三维矩阵，无法获取两个时间点中间的准确哈密顿量，所以最好选用函数表示哈密顿量
function MasterEq(ρ0, H, Ls, t, ħ=1)
    n = size(ρ0, 1)
    ρ = Array{ComplexF64}(undef, n, n, length(t))
    ρ[:, :, 1] = copy(ρ0)
    dt = t[2] - t[1]
    ndims(Ls) == 2 && (Ls = reshape(Ls, n, n, 1))
    function rhs(t_param, dm, i)
        H_current = _get_H(H, t_param, i)
        drho = -im / ħ * (H_current * dm - dm * H_current)
        for L in eachslice(Ls, dims=3)
            drho += L * dm * L' - 0.5 * (L' * L * dm + dm * L' * L)
        end
        return drho
    end
    for i in 1:length(t)-1
        t_current = t[i]
        ρ_current = ρ[:, :, i]
        k1 = rhs(t_current, ρ_current, i)
        k2 = rhs(t_current + dt / 2, ρ_current + k1 * dt / 2, i)
        k3 = rhs(t_current + dt / 2, ρ_current + k2 * dt / 2, i)
        k4 = rhs(t_current + dt, ρ_current + k3 * dt, i)
        ρ[:, :, i+1] = ρ_current + (k1 + 2k2 + 2k3 + k4) * dt / 6
    end
    return ρ
end
# 可以考虑julia的RK4算法

function SuperLind(H::AbstractMatrix, Lind, ħ=1)
    n = size(H, 1)
    super_L = zeros(ComplexF64, n^2, n^2)
    super_L = -im / ħ * (kron(H, I(n)) - kron(I(n), transpose(H)))
    Lind == 0 && (return super_L)
    ndims(Lind) == 2 && (Lind = reshape(Lind, n, n, 1))
    for L in eachslice(Lind, dims=3)
        super_L += kron(L, conj(L))
        temp = L' * L
        super_L -= 0.5 * (kron(temp, I(n)) + kron(I(n), transpose(temp)))
    end
    return super_L
end

function _get_H(H, t_param, i)
    if H isa Function
        return H(t_param)
    elseif ndims(H) == 3
        return H[:, :, i] #产生系统误差，无法获取
    end
    return H
end

function MasterEq2(ρ0, H, Ls, t, ħ=1)
    n = size(ρ0, 1)
    n_t = length(t)
    dt = t[2] - t[1]
    ρ = zeros(ComplexF64, n^2, n_t)
    ρ[:, 1] = vec(transpose(ρ0))
    cond = H isa AbstractMatrix
    cond && (super_L = SuperLind(H, Ls, ħ))
    for i in 1:n_t-1
        if !cond
            H_current = _get_H(H, t[i], i)
            super_L = SuperLind(H_current, Ls, ħ)
        end        
        k1 = super_L * ρ[:, i]
        k2 = super_L * (ρ[:, i] + k1 * dt / 2)
        k3 = super_L * (ρ[:, i] + k2 * dt / 2)
        k4 = super_L * (ρ[:, i] + k3 * dt)
        ρ[:, i+1] = ρ[:, i] + (k1 + 2k2 + 2k3 + k4) * dt / 6
    end
    return ρ
end

function MasterEq_FSL(ρ0, H, Ls, t, ħ=1)
    n = size(ρ0, 1)
    n_t = length(t)
    dt = t[2] - t[1]
    ρ = zeros(ComplexF64, n^2, n_t)
    ρ[:, 1] = vec(transpose(ρ0))
    if H isa Matrix
        H_reform = SuperLind(H, Ls, ħ)
        DR, PR = eigen(H_reform)
        PL = inv(PR)
        c = PL * ρ[:, 1]
        exp_factors = exp.(DR * t')
        ρ = PR * (c .* exp_factors)
        return ρ
    else
        dt = t[2] - t[1]
        for i in 1:n_t-1
            H isa Function ? tmp = H(t[i]) : tmp = H[:, :, i]
            H_reform = SuperLind(tmp, Ls, ħ)
            DR, PR = eigen(H_reform)
            PL = inv(PR)
            c = PL * ρ[:, i]
            exp_factors = exp.(DR * dt)
            ρ[:, i+1] = PR * (c .* exp_factors)
        end
        return ρ
    end
end
