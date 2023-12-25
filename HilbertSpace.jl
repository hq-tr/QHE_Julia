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

function bilayerhilbertspace(N_el::Int, N_orb::Int; output_type="Binary",combine_layer=false) 
    # Generate full Hilbert space for bilayer system with N_el electrons and N_orb orbitals on each layer
    basis_single_layer = fullhilbertspace(N_el, N_orb;output_type = output_type)
    if combine_layer
        basis_bilayer =[]
        for vec1 in basis_single_layer, vec2 in basis_single_layer
            push!(basis_bilayer, vcat(vec1, vec2))
        end
    else
        layer1 = []
        layer2 = []
        for vec1 in basis_single_layer, vec2 in basis_single_layer
            push!(layer1, vec1)
            push!(layer2, vec2)
        end
        basis_bilayer = [layer1, layer2]
    end
    return basis_bilayer
end

function bilayerhilbertspace(N_el1::Int, N_el2::Int,N_orb::Int; output_type="Binary",combine_layer=false) 
    # Generate full Hilbert space for bilayer system with N_orb orbitals on each layer, N_el1 electrons on layer 1 and N_el2 electrons on layer 2
    basis_layer1 = fullhilbertspace(N_el1, N_orb;output_type = output_type)
    basis_layer2 = fullhilbertspace(N_el2, N_orb;output_type = output_type)
    if combine_layer
        basis_bilayer =[]
        for vec1 in basis_layer1, vec2 in basis_layer2
            push!(basis_bilayer, vcat(vec1, vec2))
        end
    else
        layer1 = []
        layer2 = []
        for vec1 in basis_layer1, vec2 in basis_layer2
            push!(layer1, vec1)
            push!(layer2, vec2)
        end
        basis_bilayer = [layer1, layer2]
    end
    return basis_bilayer
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

export fullhilbertspace, LECspace, LECType,bilayerhilbertspace
end