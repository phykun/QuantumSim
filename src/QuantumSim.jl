module QuantumSim

using LinearAlgebra
using SparseArrays # PhononOperators.jl 中用到了

# 导出常量
include("CONST.jl")
export σ_x, σ_y, σ_z, σ_p, σ_n

# 导出物理和多体系统函数
include("PHY.jl")
export H_kinetic, Hk_period, propagation, RWA, RearrangeHam

include("ManyBody.jl")
using .ManyBody
export BoseSystem, H_b_onebody, H_b_twobody, H_BoseHubbard, H_f_onebody
export FermiSystem, H_f_onebody, H_f_twobody
export print_basis

# 导出主方程求解器
include("MasterEquation.jl")
export MasterEq, MasterEq_Diag, SuperLind

# 导出周期势求解相关函数
include("Solve_period_well.jl")
export u_nk_Position, get_bond, Blo_Wan_FBZ

# 加载并导出子模块 PhononOperators 的内容
include("PhononOperators.jl")
using .PhononOperators
export FockSpace, create, destroy, number_op, basis_state

end # module QuantumSim