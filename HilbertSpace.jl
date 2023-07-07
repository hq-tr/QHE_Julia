module HilbertSpaceGenerator
include("Misc.jl")
using .MiscRoutine

using Combinatorics

function fullhilbertspace(N_el::Int, N_orb::Int; output_type="Binary")
    list_generator = combinations(0:(N_orb-1), N_el)
    if output_type=="Binary"
        return [dex2bin(thing, N_orb) for thing in list_generator]
    elseif output_type=="Index"
        return collect(list_generator)
    elseif output_type=="Decimal"
        println("Decimal format not supported yet.")
        return
    end
end

LECType = Tuple{Int, Int, Int}
function checkLEC(state::BitVector, LEC::LECType, bothends = false)
    if bothends
        return (count(state[1:LEC[1]])<=LEC[2]) || (count(state[end-LEC[1]:end])<=LEC[2])
    else
        return count(state[1:LEC[1]])<=LEC[2]
    end
end

findLZ(state::BitVector, S::Real) = sum(state.*(-S:1:S))
findLZ(state::BitVector) = sum(state.*(0:(length(state)-1)))

findLZ(state::Vector{Int}, S::Real) = -S*length(state) + sum(state)

function fullhilbertspace(N_el::Int, N_orb::Int, L_z::Int; output_type="Binary")
    S = 0.5(N_orb-1)
    list_generator = combinations(0:(N_orb-1), N_el)
    if output_type=="Binary"
        return [dex2bin(thing, N_orb) for thing in list_generator if findLZ(thing,S)==L_z]
    elseif output_type=="Index"
        return [thing for thing in list_generator if findLZ(thing, S) == L_z]
    elseif output_type=="Decimal"
        println("Decimal format not supported yet.")
        return
    end
end

function LECspace(N_el::Int, N_orb::Int, L_z::Int, condition::LECType, bothends = false) # Output type is always binary for now
    fullspace = fullhilbertspace(N_el, N_orb, L_z)
    return [vec for vec in fullspace if checkLEC(vec, condition, bothends)]
end

function LECspace(N_el::Int, N_orb::Int, L_z::Int, conditions::Vector{LECType}) # Output type is always binary for now
    fullspace = fullhilbertspace(N_el, N_orb, L_z)
    return [vec for vec in fullspace if any(checkLEC(vec, condition) for condition in conditions)]
end

export fullhilbertspace, LECspace, LECType
end