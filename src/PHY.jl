export H_kinetic, Hk_period, propagation, RWA, RearrangeHam
function H_kinetic(z, mass, ħ)
    dz = z[2] - z[1]
    coe = (ħ / dz)^2 / (2 * mass)
    num = size(z, 1)
    H = zeros(num, num)
    for i = 1:num
        H[i, i] = 2 * coe # expendable(only 1-D)
        i != 1 && (H[i, i-1] = -coe)
        i != num && (H[i, i+1] = -coe)
    end
    return H
end

function Hk_period(z, mass, ħ)
    H = H_kinetic(z, mass, ħ)
    H[1, end] = H[1, 2]
    H[end, 1] = H[1, 2]
    return H
end

function propagation(H, psi0, t, ħ=1)
    n = size(psi0, 1)
    n_t = length(t)
    psi = zeros(ComplexF64, n, n_t)
    psi[:, 1] = psi0
    if H isa AbstractMatrix
        eig = eigen(H)
        P = eig.vectors
        D = eig.values
        c = P' * psi[:, 1]
        exp_factors = exp.(-im * D * t' / ħ)
        psi = P * (c .* exp_factors)
        return psi, P, D
    else
        dt = t[2] - t[1]
        for i in 1:n_t-1
            H isa Function ? tmp = H(t[i]) : tmp = H[:, :, i]
            eig = eigen(tmp)
            P = eig.vectors
            D = eig.values
            c = P' * psi[:, i]
            exp_factors = exp.(-im * D * dt / ħ)
            psi[:, i+1] = P * (c .* exp_factors)
        end
        return psi
    end
end

# 该函数只能处理最简单的情况，好在可以自动确定幺正变换
# 然而对角项的值可能是不恰当的，需要手动平移
# 还是不适合处理双共振问题，我不可能一直做旋波近似，这个时候可能需要代数处理
function RWA(H, ω_Ham, ħ=1)
    n = size(H, 1)
    indices = findall(x -> x != 0, ω_Ham)
    n_ω = length(indices) / 2
    @assert n_ω == n - 1 "只支持耦合项个数等于n-1"
    A = zeros(n - 1, n - 1)
    B = zeros(n - 1)
    k = 1
    for idx in indices
        i, j = idx.I
        if j > i
            A[k, i] = 1
            j != n && (A[k, j] = -1)
            B[k] = -ω_Ham[i, j]
            k += 1
        end
    end
    Uni = A \ B
    Uni = [Uni; 0]
    H_eff = H / 2 - ħ * Diagonal(Uni) + Diagonal(diag(H) / 2)
    H_eff -= ħ * ω_Ham[indices[1]] * I(n)
    return H_eff
end

function exchange_basis(Isite, Fsite, basis)
    temp = basis[Isite]
    basis[Isite] = basis[Fsite]
    basis[Fsite] = temp
    return basis
end

function exchange_H(Isite, Fsite, H)
    temp1 = H[Isite, :]
    H[Isite, :] = H[Fsite, :]
    H[Fsite, :] = temp1
    # the order make sense
    temp2 = H[:, Isite]
    H[:, Isite] = H[:, Fsite]
    H[:, Fsite] = temp2
    return H
end

function RearrangeHam(Hp::Matrix, basisp)
    H = copy(Hp)
    basis = copy(basisp)
    Ns = size(H, 1)
    for i in 1:Ns-2
        site1 = findall(H[i, i+1:end] .!= 0)
        site2 = findall(H[i, i+1:end] .== 0)
        N1 = length(site1)
        N2 = length(site2)
        if N1 == 0 || N2 == 0
            continue
        end
        while site2[1] < site1[end]
            basis = exchange_basis(site1[end] + i, site2[1] + i, basis)
            H = exchange_H(site1[end] + i, site2[1] + i, H)
            site1 = findall(H[i, i+1:end] .!= 0)
            site2 = findall(H[i, i+1:end] .== 0)
        end
    end
    return H, basis
end