module ManyBody

using LinearAlgebra

export BoseSystem, ChainLattice, H_b_onebody, H_b_twobody, H_BoseHubbard
export FermiSystem

struct BoseSystem
    N_b::Int           # 总粒子数
    N_site::Int        # 格点数
    N_max::Int         # 每个格点的最大总粒子数
    dis_spin::Vector{Int} # 自旋组分数
    basis::Vector{Vector{Vector{Int}}}       # 基矢
    basis_dict::Dict{Vector{Vector{Int}},Int}# 基矢字典
    function BoseSystem(N_b::Int, N_site::Int)
        dis_spin = [N_b]
        N_max = N_b
        basis = _boson_basis(N_b, N_site)
        basis_dict = Dict(state => idx for (idx, state) in enumerate(basis))
        return new(N_b, N_site, N_max, dis_spin, basis, basis_dict)
    end
    function BoseSystem(N_b::Int, N_site::Int, N_max::Int)
        dis_spin = [N_b]
        basis = _boson_basis(N_b, N_site, N_max)
        basis_dict = Dict(state => idx for (idx, state) in enumerate(basis))
        return new(N_b, N_site, N_max, dis_spin, basis, basis_dict)
    end
    function BoseSystem(N_b::Int, N_site::Int, N_max::Int, dis_spin::Vector{Int})
        basis = _boson_basis(N_b, N_site, N_max, dis_spin)
        basis_dict = Dict(state => idx for (idx, state) in enumerate(basis))
        return new(N_b, N_site, N_max, dis_spin, basis, basis_dict)
    end
end

function _boson_basis(N_b::Int, N_site::Int)
    @assert N_b ≥ 1 "玻色子数目应大于等于1"
    @assert N_site ≥ 1 "格点个数应该大于等于1"
    dis = zeros(Int, N_site)
    dis[1] = N_b
    Ns = binomial(N_b + N_site - 1, N_b)
    basis = Vector{Vector{Int}}(undef, Ns)
    basis[1] = copy(dis)
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
        basis[i] = copy(dis)
    end
    return [[state] for state in basis]
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
    basis = Vector{Vector{Int}}(undef, Ns)
    basis[1] = copy(dis)
    N_num = 2
    while dis != reverse(basis[1])
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
            basis[N_num] = copy(dis)
            N_num += 1
        end
    end
    return [[state] for state in basis]
end

function _boson_basis(N_b::Int, N_site::Int, N_max::Int, dis_spin::Vector{Int})
    @assert sum(dis_spin) == N_b "总自旋数应该等于玻色子数目"
    N_spin = length(dis_spin)
    N_spin == 1 && return _boson_basis(dis_spin[1], N_site, N_max)
    spin_basis = Vector{Vector{Vector{Int}}}(undef, N_spin)
    for i in 1:N_spin
        tmp = _boson_basis(dis_spin[i], N_site, N_max)
        spin_basis[N_spin+1-i] = [state[1] for state in tmp]
    end
    basis = Vector{Vector{Vector{Int}}}()
    all_combinations = Iterators.product(spin_basis...)
    for combo in all_combinations
        valid = true
        for site in 1:N_site
            total = sum(spin_config[site] for spin_config in combo)
            if total > N_max
                valid = false
                break
            end
        end
        valid && push!(basis, collect(combo))
    end
    return reverse.(basis)
end

function ChainLattice(N_site, periodic=false)
    cond = periodic && N_site > 2
    hopping_pairs = [(i, i + 1) for i in 1:N_site-1]
    cond && push!(hopping_pairs, (N_site, 1))
    for i in 1:N_site-1
        push!(hopping_pairs, (i + 1, i))
    end
    cond && push!(hopping_pairs, (1, N_site))
    return hopping_pairs
end

function H_b_onebody(system::BoseSystem, t, tuples)
    isa(t, Number) && (t = fill(t, length(tuples)))
    basis = system.basis
    basis_dict = system.basis_dict
    N_max = system.N_max
    N_spin = length(system.dis_spin)
    Ns = length(basis)
    H = zeros(Ns, Ns)
    for idx_spin = 1:N_spin
        for (idx1, (i, j)) in enumerate(tuples)
            for (idx2, state) in enumerate(basis)
                state_spin = copy(state[idx_spin])
                if state_spin[j] >= 1 && state_spin[i] < N_max
                    state_s_f = copy(state_spin)
                    state_s_f[j] -= 1
                    state_s_f[i] += 1
                    coe = t[idx1] * sqrt(state_spin[j]) * sqrt(state_s_f[i])
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

function H_b_twobody(system::BoseSystem, U, quadruples)
    isa(U, Number) && (U = fill(U, length(quadruples)))
    @assert length(U) == length(quadruples) "相互作用强度与二体数量不匹配"
    basis = system.basis
    basis_dict = system.basis_dict
    N_max = system.N_max
    N_spin = length(system.dis_spin)
    Ns = length(basis)
    H = zeros(Ns, Ns)
    for idx_spin = 1:N_spin
        for (idx1, (i, j, k, l)) in enumerate(quadruples)
            for (idx2, state) in enumerate(basis)
                state_spin = copy(state[idx_spin])
                if state_spin[k] >= 1 && state_spin[l] >= 1
                    state_s_t = copy(state_spin)
                    state_s_t[k] -= 1
                    state_s_t[l] -= 1
                    state_s_t[k] < 0 && continue
                    if state_s_t[i] < N_max && state_s_t[j] < N_max
                        state_s_f = copy(state_s_t)
                        state_s_f[i] += 1
                        state_s_f[j] += 1
                        state_s_f[i] > N_max && continue
                        coe = U[idx1] / 2 *
                              sqrt(state_spin[l]) * sqrt(state_s_t[k] + 1) *
                              sqrt(state_s_t[j] + 1) * sqrt(state_s_f[i])
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

function H_BoseHubbard(system::BoseSystem, t, U, periodic=false)
    tuples = ChainLattice(system.N_site, periodic)
    quadruples = [(i, i, i, i) for i in 1:system.N_site]
    return H_b_onebody(system, t, tuples) + H_b_twobody(system, U, quadruples)
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

end
