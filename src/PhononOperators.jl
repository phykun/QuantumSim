module PhononOperators

using SparseArrays
using LinearAlgebra

export FockSpace, create, destroy, number_op, basis_state

"""
    FockSpace(modes::Int, cutoff::Int)

Defines the Hilbert space for a multi-mode bosonic system.

# Fields
- `modes::Int`: Number of bosonic modes.
- `cutoff::Int`: Maximum number of particles allowed in any single mode. 
      The Fock state for each mode is represented by a vector of length `cutoff + 1`.
- `dim::Int`: The total dimension of the Hilbert space, calculated as `(cutoff + 1)^modes`.
"""
struct FockSpace
    modes::Int
    cutoff::Int
    dim::Int

    function FockSpace(modes::Int, cutoff::Int)
        dim = (cutoff + 1)^modes
        new(modes, cutoff, dim)
    end
end

# Helper function to get the single-mode identity matrix
_identity(cutoff::Int) = spdiagm(0 => ones(cutoff + 1))

"""
    create(fs::FockSpace, mode::Int)

Creates a sparse matrix representing the creation operator (â†) for the specified mode.
"""
function create(fs::FockSpace, mode::Int)
    1 <= mode <= fs.modes || throw(ArgumentError("Mode is out of bounds."))
    a_dag_single = spdiagm(-1 => [sqrt(n) for n in 1:fs.cutoff])
    fs.modes == 1 && return a_dag_single
    ops = [_identity(fs.cutoff) for _ in 1:fs.modes]
    ops[mode] = a_dag_single
    return kron(ops...)
end

"""
    destroy(fs::FockSpace, mode::Int)

Creates a sparse matrix representing the annihilation operator (â) for the specified mode.
"""
function destroy(fs::FockSpace, mode::Int)
    1 <= mode <= fs.modes || throw(ArgumentError("Mode is out of bounds."))
    a_single = spdiagm(1 => [sqrt(n) for n in 1:fs.cutoff])
    fs.modes == 1 && return a_single
    ops = [_identity(fs.cutoff) for _ in 1:fs.modes]
    ops[mode] = a_single
    return kron(ops...)
end

"""
    number_op(fs::FockSpace, mode::Int)

Creates a sparse matrix representing the number operator (n̂ = â†â) for the specified mode.
"""
function number_op(fs::FockSpace, mode::Int)
    1 <= mode <= fs.modes || throw(ArgumentError("Mode is out of bounds."))
    n_single = spdiagm(0 => [Float64(n) for n in 0:fs.cutoff])
    fs.modes == 1 && return n_single
    ops = [_identity(fs.cutoff) for _ in 1:fs.modes]
    ops[mode] = n_single
    return kron(ops...)
end

"""
    basis_state(fs::FockSpace, occupations::Vector{Int})

Creates a state vector corresponding to a specific Fock state.
"""
function basis_state(fs::FockSpace, occupations::Vector{Int})
    length(occupations) == fs.modes || throw(ArgumentError("Number of occupations must match number of modes."))
    any(occ -> occ > fs.cutoff, occupations) && throw(ArgumentError("Occupation number cannot exceed cutoff."))
    index = 1
    for i in 1:fs.modes
        index += occupations[i] * (fs.cutoff + 1)^(i - 1)
    end
    ψ = spzeros(ComplexF64, fs.dim)
    ψ[index] = 1.0
    return Vector(ψ)
end

end # module
