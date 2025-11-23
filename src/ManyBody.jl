module ManyBody

using LinearAlgebra

export BoseSystem, print_basis, H_b_onebody, H_b_twobody, H_BoseHubbard, H_f_onebody
export FermiSystem

struct BoseSystem
    N_b::Int           # 总粒子数
    N_site::Int        # 格点数
    N_max::Int         # 每个格点的最大总粒子数
    N_spin::Int        # 自旋组分数
    basis::Matrix{Int}     # 基矢
    function BoseSystem(N_b::Int, N_site::Int)
        N_spin = 1
        N_max = N_b
        basis = _boson_basis(N_b, N_site)
        return new(N_b, N_site, N_max, N_spin, basis)
    end
    function BoseSystem(N_b::Int, N_site::Int, N_max::Int)
        N_spin = 1
        basis = _boson_basis(N_b, N_site, N_max)
        return new(N_b, N_site, N_max, N_spin, basis)
    end
    function BoseSystem(N_b::Int, N_site::Int, N_max::Int, dis_spin::Vector{Int})
        N_spin = length(dis_spin)
        basis = _boson_basis(N_b, N_site, N_max, dis_spin)
        return new(N_b, N_site, N_max, N_spin, basis)
    end
    function BoseSystem(N_b::Int, N_site::Int, N_max::Int, N_spin::Int)
        basis = _boson_basis(N_b, N_site, N_max, N_spin)
        return new(N_b, N_site, N_max, N_spin, basis)
    end
end

function _boson_basis(N_b::Int, N_site::Int)
    N_b == 0 && return zeros(Int, N_site)
    @assert N_site ≥ 1 "格点个数应该大于等于1"
    dis = zeros(Int, N_site)
    dis[1] = N_b
    Ns = binomial(N_b + N_site - 1, N_b)
    basis = zeros(Int, N_site, Ns)
    basis[:, 1] = copy(dis)
    for i in 2:Ns
        indices = findall(!=(0), dis)
        if indices[end] != N_site
            # 如果最右位置0填充，从最右填充位置往右边移动一个粒子
            site = indices[end]
            dis[site] -= 1
            dis[site+1] = 1
        else
            # 如果最右位置有填充，次最右填充位置往右边移动一个粒子
            site = indices[end-1]
            dis[site] -= 1
            dis[site+1] = 1 + dis[end]# 末位粒子前移
            (site != N_site - 1) && (dis[end] = 0)
            # 如果非最右倒数第二个位置，将最末尾清零
        end
        basis[:, i] = copy(dis)
    end
    return basis
end

function _boson_basis(N_b::Int, N_site::Int, N_max::Int)
    @assert N_max >= 1 "单格点最大占据数应该大于等于1"
    @assert N_max * N_site > N_b "玻色子数目超过最大装载数"
    (N_b <= N_max) && return _boson_basis(N_b, N_site)
    N_sum = floor(Int, N_b / (N_max + 1))
    Ns = 0
    for i in 0:N_sum
        tmp = (-1)^(i) * binomial(N_site, i)
        tmp = tmp * binomial(N_b + N_site - 1 - i * (N_max + 1), N_site - 1)
        Ns += tmp
    end
    N_ini = floor(Int, N_b / N_max)
    dis = zeros(Int, N_site)
    dis[1:N_ini] .= N_max
    dis[N_ini+1] = N_b % N_max
    basis = zeros(Int, N_site, Ns)
    basis[:, 1] = copy(dis)
    N_num = 2
    while dis != reverse(basis[:, 1])
        indices = findall(!=(0), dis)
        if indices[end] != N_site
            site = indices[end]
            dis[site] -= 1
            dis[site+1] = 1
        else
            site = indices[end-1]
            dis[site] -= 1
            dis[site+1] = 1 + dis[end]
            (site != N_site - 1) && (dis[end] = 0)
        end
        if all(x -> x <= N_max, dis)
            basis[:, N_num] = copy(dis)
            N_num += 1
        end
    end
    return basis
end

function _boson_basis(N_b::Int, N_site::Int, N_max::Int, dis_spin::Vector{Int})
    @assert sum(dis_spin) == N_b "总自旋数应该等于玻色子数目"
    N_spin = length(dis_spin)
    N_spin == 1 && return _boson_basis(dis_spin[1], N_site, N_max)
    spin_basis = _boson_basis.(dis_spin, N_site, N_max)
    spin_Ns = [size(b, 2) for b in spin_basis]
    indices = Iterators.product([1:n for n in reverse(spin_Ns)]...)
    temp_col = Vector{Int}(undef, N_site * N_spin)
    occupancy = Vector{Int}(undef, N_spin)
    valid_col = Vector{Vector{Int}}()
    @inbounds for tup in indices
        for spin_idx in 1:N_spin
            real_idx = N_spin - spin_idx + 1
            row_start = (spin_idx - 1) * N_site + 1
            row_end = spin_idx * N_site
            col_idx = tup[real_idx]
            temp_col[row_start:row_end] = spin_basis[spin_idx][:, col_idx]
        end
        valid = true
        for i in 1:N_spin
            occupancy[i] = sum(@view temp_col[i:N_site:end])
            if occupancy[i] > N_max
                valid = false
                break
            end
        end
        valid && push!(valid_col, copy(temp_col))
    end
    return hcat(valid_col...)
end

function _boson_basis(N_b::Int, N_site::Int, N_max::Int, N_spin::Int)
    N_all = binomial(N_b + N_spin - 1, N_b)
    all_dis_spin = _boson_basis(N_b, N_spin)
    basis = Matrix{Int}(undef, N_site * N_spin, 0)
    for i in 1:N_all
        dis_spin = all_dis_spin[:, i]
        tmp_basis = _boson_basis(N_b, N_site, N_max, dis_spin)
        basis = hcat(basis, tmp_basis)
    end
    return basis
end

function print_basis(system::BoseSystem)
    Ns = size(system.basis, 2)
    N_site = system.N_site
    N_spin = system.N_spin
    basis = system.basis
    println("基矢共有 $Ns 个")
    @inbounds for i in 1:Ns
        println("基矢$i:")
        for j in 1:N_spin
            n_start = (j - 1) * N_site + 1
            n_end = j * N_site
            print("自旋$j:", basis[n_start:n_end, i],";")
        end
        println()
    end
end

function H_b_onebody(system::BoseSystem, t::Matrix)
    N_site = system.N_site
    N_spin = system.N_spin
    @assert size(t, 1) == N_site "格点维度不匹配"
    basis = system.basis
    N_max = system.N_max
    Ns = size(basis, 2)
    H = zeros(Ns, Ns)
    for idx_spin = 1:N_spin
        for i in axes(t, 1), j in axes(t, 2)
            for (idx, state) in enumerate(eachcol(basis))
                s1 = (idx_spin - 1) * N_site + i
                s2 = (idx_spin - 1) * N_site + j
                if state[s2] >= 1 && state[s1] < N_max
                    state_f = copy(state)
                    state_f[s2] -= 1
                    state_f[s1] += 1
                    coe = -t[i,j] * sqrt(state[j]) * sqrt(state_f[i])
                    idx_f = findfirst(col -> state_f == col, eachcol(basis))
                    !isnothing(idx_f) && (H[idx_f,idx] += coe)
                end
            end
        end
    end
    @assert H==H' "Hamiltonian is not hermitian"
    return H
end

# Tight-bonding situation
function H_b_onebody(system::BoseSystem, t::Number, periodic::Bool)
    hopping = zeros(N_site,N_site)
    for i in 1:N_site
        i!=1 && (hopping[i,i-1]=t)
        i!=N_site && (hopping[i,i+1]=t)
    end
    if periodic
        hopping[1,N_site] = t
        hopping[N_site,1] = t
    end
    return H_b_onebody(system,hopping)
end

# t::{N_site,N_site,N_spin,N_spin} only for first energy bond
function H_b_onebody(system::BoseSystem, t::Array{Float64, 4})
    N_site = system.N_site
    N_spin = system.N_spin
    @assert size(t, 1) == N_site "格点维度不匹配"
    @assert size(t, 3) == N_spin "自旋维度不匹配"
    basis = system.basis
    N_max = system.N_max
    Ns = size(basis, 2)
    H = zeros(Ns, Ns)
    for index in CartesianIndices(t)
        i, j, spin1, spin2 = index.I
        for (idx, state) in enumerate(eachcol(basis))
            s1 = (spin1 - 1) * N_site + i
            s2 = (spin2 - 1) * N_site + j
            if state[s2] >= 1 && state[s1] < N_max
                state_f = copy(state)
                state_f[s2] -= 1
                state_f[s1] += 1
                coe = -t[i,j,spin1,spin2] * sqrt(state[j]) * sqrt(state_f[i])
                idx_f = findfirst(col -> state_f == col, eachcol(basis))
                !isnothing(idx_f) && (H[idx_f,idx] += coe)
            end
        end
    end
    @assert H==H' "Hamiltonian is not hermitian"
    return H
end

# onsite
function H_b_twobody(system::BoseSystem, U::Number)
    N_site = system.N_site
    N_spin = system.N_spin
    basis = system.basis
    Ns = size(basis,2)
    H = zeros(Ns, Ns)
    for spin1 in 1:N_spin, spin2 in 1:N_spin
        for index in 1:N_site
            for (idx, state) in enumerate(eachcol(basis))
                s1 = (spin1 - 1) * N_site + index
                s2 = (spin2 - 1) * N_site + index
                if state[s1] >= 1 && state[s2] >= 1
                    if spin1 == spin2
                        coe = U / 2 * state[s1] * (state[s1]-1)
                    else
                        coe = U / 2 * state[s1] * state[s2]
                    end
                    H[idx,idx] += coe
                end
            end
        end
    end
    @assert H==H' "Hamiltonian is not hermitian"
    return H
end

function H_b_twobody(system::BoseSystem, U::Array{Float64, 4})
    N_site = system.N_site
    @assert size(U,1) == N_site "格点维度不匹配"
    N_max = system.N_max
    N_spin = system.N_spin
    basis = system.basis
    Ns = size(basis,2)
    H = zeros(Ns, Ns)
    for spin1 in 1:N_spin, spin2 in 1:N_spin
        for index in CartesianIndices(U)
            i,j,k,l = index.I
            s_i = (spin1 - 1) * N_site + i
            s_j = (spin2 - 1) * N_site + j
            s_k = (spin1 - 1) * N_site + k
            s_l = (spin2 - 1) * N_site + l
            for (idx, state) in enumerate(eachcol(basis))
                if state[s_k] >= 1 && state[s_l] >= 1
                    state_t = copy(state)
                    state_t[s_k] -= 1
                    state_t[s_l] -= 1
                    state_t[s_k] < 0 && continue
                    if state_t[s_i] < N_max && state_t[s_j] < N_max
                        state_f = copy(state_t)
                        state_f[s_i] += 1
                        state_f[s_j] += 1
                        state_f[s_i] > N_max && continue
                        coe = U[i,j,k,l] / 2 *
                              sqrt(state[s_k]) * sqrt(state_t[s_l] + 1) *
                              sqrt(state_t[s_j] + 1) * sqrt(state_f[s_i])
                        idx_f = findfirst(col -> state_f == col, eachcol(basis))
                        !isnothing(idx_f) && (H[idx_f,idx] += coe)
                    end
                end
            end
        end
    end
    @assert H==H' "Hamiltonian is not hermitian"
    return H
end

function H_BoseHubbard(system::BoseSystem, t::Number, U::Number, periodic=false)
    return H_b_onebody(system, t, periodic) + H_b_twobody(system, U)
end

struct FermiSystem
    N_f::Int           # 总粒子数
    N_site::Int        # 格点数
    dis_spin::Vector{Int} # 自旋组分数
    basis::Vector{Vector{Vector{Int}}}      # 基矢
    basis_dict::Dict{Vector{Vector{Int}},Int}
    function FermiSystem(N_f::Int, N_site::Int, dis_spin::Vector{Int64})
        basis = _fermi_basis(N_f, N_site, dis_spin)
        basis_dict = Dict(state => idx for (idx, state) in enumerate(basis))
        return new(N_f, N_site, dis_spin, basis, basis_dict)
    end
    function FermiSystem(N_f::Int, N_site::Int)
        dis_spin = [N_f]
        basis = _fermi_basis(N_f, N_site)
        basis_dict = Dict(state => idx for (idx, state) in enumerate(basis))
        return new(N_f, N_site, dis_spin, basis, basis_dict)
    end
end

function _fermi_basis(N_f::Int, N_site::Int)
    @assert N_f ≥ 0 && N_site ≥ N_f "参数需满足 0 ≤ N_f ≤ N_site"
    Ns = binomial(N_site, N_f)
    dis = vcat(ones(Int, N_f), zeros(Int, N_site - N_f))
    basis = Vector{Vector{Int}}(undef, Ns)
    basis[1] = copy(dis)
    for i in 2:Ns
        ones_pos = findall(==(1), dis)
        last_one = ones_pos[end]
        if last_one != N_site
            dis[last_one] = 0
            dis[last_one+1] = 1
        else
            pivot = N_site
            while dis[pivot-1] == 1
                pivot -= 1
            end
            N_end = N_site - pivot + 1
            site = ones_pos[N_f-N_end]
            dis[site] = 0
            dis[pivot:end] .= 0
            dis[site+1:site+N_end+1] .= 1
        end
        basis[i] = copy(dis)
    end
    return [[state] for state in basis]
end

function _fermi_basis(N_f::Int, N_site::Int, dis_spin::Vector{Int64})
    @assert sum(dis_spin) == N_f "参数需满足：总自旋数等于费米子数目"
    N_spin = length(dis_spin)
    spin_basis = Vector{Vector{Vector{Int}}}(undef, N_spin)
    for i in 1:N_spin
        tmp = _fermi_basis(dis_spin[i], N_site)
        spin_basis[N_spin+1-i] = [state[1] for state in tmp]
    end
    basis = Vector{Vector{Vector{Int}}}()
    all_combinations = Iterators.product(spin_basis...)
    for combo in all_combinations
        push!(basis, collect(combo))
    end
    return reverse.(basis)
end

function H_f_onebody(system::FermiSystem, t, tuples)
    isa(t, Number) && (t = fill(t, length(tuples)))
    basis = system.basis
    basis_dict = system.basis_dict
    N_spin = length(system.dis_spin)
    Ns = length(basis)
    H = zeros(Ns, Ns)
    for idx_spin = 1:N_spin
        for (idx1, (i, j)) in enumerate(tuples)
            for (idx2, state) in enumerate(basis)
                state_spin = copy(state[idx_spin])
                if state_spin[j] == 1 && state_spin[i] == 0
                    state_s_f = copy(state_spin)
                    coe1 = sum(state_s_f[1:j-1] .== 1)
                    state_s_f[j] -= 1
                    coe2 = sum(state_s_f[1:i-1] .== 1)
                    state_s_f[i] += 1
                    coe = t[idx1] * (-1)^(coe1 + coe2)
                    state_f = copy(state)
                    state_f[idx_spin] = state_s_f
                    if haskey(basis_dict, state_f)
                        new_idx = basis_dict[state_f]
                        H[idx2, new_idx] += coe
                    end
                end
            end
        end
    end
    return H
end

function H_f_twobody(system::FermiSystem, U, quadruples)
    isa(U, Number) && (U = fill(U, length(quadruples)))
    @assert length(U) == length(quadruples) "相互作用强度与二体数量不匹配"
    basis = system.basis
    basis_dict = system.basis_dict
    N_spin = length(system.dis_spin)
    Ns = length(basis)
    H = zeros(Ns, Ns)
    for idx_spin = 1:N_spin
        for (idx1, (i, j, k, l)) in enumerate(quadruples)
            for (idx2, state) in enumerate(basis)
                state_spin = copy(state[idx_spin])
                if state_spin[k] == 1 && state_spin[l] == 1
                    state_s_f = copy(state_spin)
                    coe_s = sum(state_s_f[1:k-1] .== 1)
                    state_s_f[k] -= 1
                    state_s_f[l] -= 1
                    state_s_f[k] < 0 && continue
                    coe_s += sum(state_s_f[1:l-1] .== 1)
                    if state_s_t[i] == 0 && state_s_t[j] == 0
                        coe_s += sum(state_s_f[1:j-1] .== 1)
                        state_s_f[j] += 1
                        coe_s += sum(state_s_f[1:i-1] .== 1)
                        state_s_f[i] += 1
                        state_s_f[i] > 1 && continue
                        coe = U[idx1] / 2 * (-1)^(coe_s)
                        state_f = copy(state)
                        state_f[idx_spin] = state_s_f
                        if haskey(basis_dict, state_f)
                            final_idx = basis_dict[state_f]
                            H[idx2, final_idx] += coe
                        end
                    end
                end
            end
        end
    end
    return H
end

function H_FermiHubbard(system::FermiSystem, t, U, periodic=false)
    tuples = ChainLattice(system.N_site, periodic)
    quadruples = [(i, i, i, i) for i in 1:system.N_site]
    return H_f_onebody(system, t, tuples) + H_f_twobody(system, U, quadruples)
end

end
