module FCI_states
using LinearAlgebra
import Base.+, Base.*
import LinearAlgebra.⋅
import Base.display

include("Misc.jl")
using .MiscRoutine

include("FQH_state_v2.jl")
using .FQH_states

import .FQH_states.readwf, .FQH_states.printwf

abstract type  AbstractFCI_state <: AbstractFQH_state end

struct FCI_state <: AbstractFCI_state
    basis::Vector{BitVector}
    coef::Vector{Number}
    Nx::Int
    Ny::Int
    FCI_state(basis,coef,Nx,Ny) = begin
    	@assert length(basis) == length(coef) "The specified basis and coefficient do not have the same length!\nBasis has $(length(basis)) element(s) but coefficient vector has $(length(coef)) element(s)."
    	@assert length(basis[1]) == Nx * Ny "The specified basis must have Nx * Ny orbital!\nBasis has $(length(basis)) orbital(s) but Nx*Ny = $(Nx*Ny)."
    	return new(basis,coef,Nx,Ny)
    end
end


mutable struct FCI_state_mutable <: AbstractFCI_state
    basis::Vector{BitVector}
    coef::Vector{Number}
    Nx::Int
    Ny::Int
    FCI_state_mutable(basis,coef,Nx,Ny) = begin
    	@assert length(basis) == length(coef) "The specified basis and coefficient do not have the same length!\nBasis has $(length(basis)) element(s) but coefficient vector has $(length(coef)) element(s)."
    	@assert length(basis[1]) == Nx * Ny "The specified basis must have Nx * Ny orbital!\nBasis has $(length(basis)) orbital(s) but Nx*Ny = $(Nx*Ny)."
    	return new(basis,coef,Nx,Ny)
    end
end

# ------------Input/Output
function printwf_bin(state::AbstractFCI_state; fname = "")
    D = dim(state)
    if length(fname) == 0
        println("$D $(state.Nx) $(state.Ny)")
        for i in 1:D
            println(prod(string.((Int.(state.basis[i])))))
            println(replace("$(state.coef[i])", "im"=>"j", " "=>""))
        end
    else
        open(fname,"w") do f
            write(f,"$D $(state.Nx) $(state.Ny)\n")
            for i in 1:D
                writebasis = prod(string.((Int.(state.basis[i]))))
                writecoef  = replace("$(state.coef[i])", "im"=>"j", " "=>"")
                write(f,"$writebasis\n$writecoef\n")
            end 
        end

    end
end

function printwf_dec(state::AbstractFCI_state; fname = "")
    D = dim(state)
    if length(fname) == 0
        println("$D $(state.Nx) $(state.Ny)")
        for i in 1:D
            println(sum(2 .^ bin2dex(state.basis[i])))
            println(replace("$(state.coef[i])", "im"=>"j", " "=>""))
        end
    else
        open(fname,"w") do f
            write(f,"$D $(state.Nx) $(state.Ny)\n")
            for i in 1:D
                writebasis = sum(2 .^ bin2dex(state.basis[i]))
                writecoef  = replace("$(state.coef[i])", "im"=>"j", " "=>"")
                write(f,"$(writebasis)\n$writecoef\n")
            end 
        end

    end
end

function printwf(state::AbstractFCI_state; fname="", format=:BIN)
    if format == :BIN
        printwf_bin(state;fname=fname)
    elseif format == :DEC
        printwf_dec(state;fname=fname)
    else
        println("Format not supported.")
    end
end


export AbstractFCI_state, FCI_state, FCI_state_mutable, printwf, readwf

end