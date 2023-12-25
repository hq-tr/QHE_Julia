module FQH_multilayer

using LinearAlgebra
import Base.+, Base.*
import LinearAlgebra.⋅
import Base.display

include("Misc.jl")
using .MiscRoutine

abstract type AbstractFQH_multilayer_state end

struct FQH_multilayer_state <: AbstractFQH_multilayer_state
	basis::Vector{Vector{BitVector}}
	coef:: Vector{Number}
	#----------- To be implemented later if necessary:
	#LLindex::Vector{Int64}
	#shift = zeros(LLindex) 
end

struct FQH_multilayer_state_mutable <: AbstractFQH_multilayer_state
	basis::Vector{Vector{BitVector}}
	coef:: Vector{Number}
	#----------- To be implemented later if necessary:
	#LLindex::Vector{Int64}
	#shift = zeros(LLindex) 
end

nlayers(state::AbstractFQH_multilayer_state) = length(state.basis[1])

export AbstractFQH_multilayer_state, FQH_multilayer_state, FQH_multilayer_state_mutable,nlayers



module SecondLLOperations
using ..FQH_multilayer

AbstractSLL_state = AbstractFQH_multilayer_state
SLL_state = FQH_multilayer_state
SLL_state_mutable = FQH_multilayer_state_mutable

function basisconvert(basis::Vector{Int64})
	layer1 = BitVector()
	layer2 = BitVector()
	for x in basis
		if x==1 
			push!(layer1,1)
			push!(layer2,0)
		elseif x==2
			push!(layer1,0)
			push!(layer2,1)
		elseif x==3
			push!(layer1,1)
			push!(layer2,1)
		else
			push!(layer1,0)
			push!(layer2,0)
		end
	return [layer1[2:end-1], layer2]
end

function basisconvert(basis::Vector{BitVector})
	return basis[1]*1 + basis[2]*2
end

function SLLreadwf(fname::String; mutable=false)
        f = open(fname)
        content = readlines(f)
        dim = parse(Int64, content[1])
        basis = [Vector(map(x->convertbasis(parse(Int64, x), split(y,"")))) for y in content[2:2:end]]
        #println(basis[1])
        println("The dimension is $dim")
        try
            global co = [parse(Float64, x) for x in content[3:2:end]]
        catch ArgumentError
            println("Reading coefficients as complex numbers")
            global co = [parse(Complex{Float64}, x) for x in content[3:2:end]]
        finally
            close(f)
            #println("Success")
        end
        if mutable
            return SLL_state_mutable(basis,co)
        else
            return SLL_state(basis, co)
        end
end

function SLLreadwf(basisname::String, coefname::String; mutable=false)
        f = open(basisname)
        content = readlines(f)
        basis = [Vector(map(x->convertbasis(parse(Int64, x), split(y,"")))) for y in content]
		close(f)
		dim = length(basis)
        #println(basis[1])
        println("The dimension is $dim")
        f = open(coefname)
        content=readlines(f)
        try
            global co = [parse(Float64, x) for x in content]
        catch ArgumentError
            println("Reading coefficients as complex numbers")
            global co = [parse(Complex{Float64}, x) for x in content]
        finally
            close(f)
            #println("Success")
        end
        if mutable
            return SLL_state_mutable(basis,co)
        else
            return SLL_state(basis, co)
        end
end

function SLLprintwf(state::AbstractSLL_state; fname = "")
    D = dim(state)
    if length(fname) == 0
        println(D)
        for i in 1:D
            display(basisconvert(state.basis[i]))
            println(replace("$(state.coef[i])", "im"=>"j", " "=>""))
        end
    else
        open(fname,"w") do f
            write(f,"$D\n")
            for i in 1:D
                writebasis = prod(string.((Int.(state.basis[i]))))
                writecoef  = replace("$(state.coef[i])", "im"=>"j", " "=>"")
                write(f,"$writebasis\n$writecoef\n")
            end 
        end

    end
end

function display(vec::AbstractSLL_state) SLLprintwf(state) end


end

export AbstractSLL_state, SLL_state, SLL_state_mutable, SLLprintwf, SLLreadwf, display
end
end