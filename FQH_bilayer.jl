# ========================================================
#
#                      BILAYER MODULE
#
# Two layers with EQUAL number of orbitals Nₒ
# Basis recorded as a single binary string of length 2Nₒ
#
#
# =========================================================
module BilayerFQH
include("FQH_multilayer.jl")
include("FQH_state_v2.jl")
include("Density.jl")
include("Misc.jl")
using .FQH_multilayer
using .FQH_states
using .ParticleDensity
using .MiscRoutine
using Plots
using SpecialFunctions
using BenchmarkTools
using SparseArrays

import Base.display
import Base.split

abstract type Abstractbilayer_state end

struct bilayer_state <: Abstractbilayer_state
    basis::Vector{Vector{BitVector}}
    coef:: Vector{Number}
    #----------- To be implemented later if necessary:
    #LLindex::Vector{Int64}
    #shift = zeros(LLindex) 
end

mutable struct bilayer_state_mutable<: Abstractbilayer_state
    basis::Vector{Vector{BitVector}}
    coef:: Vector{Number}
    #----------- To be implemented later if necessary:
    #LLindex::Vector{Int64}
    #shift = zeros(LLindex) 
end

function getdim(vec::Abstractbilayer_state)
    return lenght(vec.basis[1])
end

function split(vec::Abstractbilayer_state)
    state1 = disk_normalize(FQH_state(map(x->x[1],vec.basis), vec.coef))
    state2 = disk_normalize(FQH_state(map(x->x[2],vec.basis), vec.coef),0.5)
    return state1, state2
end

function basissplit(basis::BitVector)
    No = length(basis) ÷ 2
    return [basis[1:No], basis[No+1:end]]
end

function basiscombine(basis::Vector{BitVector})
    return vcat(basis[1], basis[2])
end

function bilayerreadwf(fname::String; mutable=false)
        f = open(fname)
        content = readlines(f)
        dim = parse(Int64, content[1])
        basis_line = [BitVector(map(x->parse(Bool, x), split(y,""))) for y in content[2:2:end]]
        basis = Vector{BitVector}[]
        push!(basis, [basissplit(vec)[1] for vec in basis_line])
        push!(basis, [basissplit(vec)[2] for vec in basis_line])
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
            return bilayer_state_mutable(basis,co)
        else
            return bilayer_state(basis, co)
        end
end

function display(vec::Abstractbilayer_state)  bilayerprintwf(vec) end

function bilayerprintwf(vec::Abstractbilayer_state;fname="",format=:BIN)
    dim = length(vec.basis[1])
    combined_basis = [vcat(vec.basis[1][i], vec.basis[2][i]) for i in 1:dim]
    state = FQH_state(combined_basis, vec.coef)
    printwf(state;fname=fname,format=format)
end

# Single-particle LLL eigenstates. Here Δ is the shift. 
single_particle_state_disk(z::Number,m::Integer,Δ::Number) = z.^(m+Δ) * exp.(-abs.(z)^2) * sqrt(2^m / (2π)) / sqrt(gamma(m+Δ+1))

single_particle_state_disk(z::Vector{T} where T <: Number,m::Integer, Δ::Number) = z.^(m+Δ) * exp.(-abs.(z)^2) * sqrt(2^m / (2π)) / sqrt(gamma(m+Δ+1))

# Calculate the electron density given a state
function disk_density(vec::Abstractbilayer_state)
    state1, state2 = split(vec)
    # Prepare the plot area
    No = length(state1.basis[1][1])
    R_max = sqrt(2*No)+0.2
    x = -R_max:0.05:R_max
    y = -R_max:0.05:R_max

    N = length(x)

    Y = repeat(y, inner=(1,N))
    X = collect(transpose(Y))
    Z = 0.5(X+Y*im)

    # Prepare the single-particle matrix
    single_particle1 = [single_particle_state_disk.(Z,m,0) for m in 0:No]
    single_particle2 = [single_particle_state_disk.(Z,m,0.5) for m in 0:No]

    # Calculate the density
    D = dim(state1)
    den = zeros(size(Z))
    @time begin
    for i in 1:D
        print("\r$i\t")
        for j in i:D
            coef = 2^(i!=j) * conj(state1.coef[i]) * state1.coef[j]
            density_element_gen!(den, coef, state1.basis[i], state1.basis[j], single_particle1)
            coef = 2^(i!=j) * conj(state2.coef[i]) * state2.coef[j]
            density_element_gen!(den, coef, state2.basis[i], state2.basis[j], single_particle2)
        end
    end
    end  # end of @time code segment
    return x,y,den
end

function disk_density(vec::Abstractbilayer_state,fname::String)
    x,y,den = disk_density(vec)

    N = length(x)

    Y = repeat(y, inner=(1,N))
    X = collect(transpose(Y))

    # Save density values to files
    open("$(fname).dat", "w") do f
        for i in 1:N, j in 1:N
            write(f, "$(X[i,j])\t$(Y[i,j])\t$(den[i,j])\n")
        end
    end

    # Plot density
    p = heatmap(x, y, den, aspect_ratio=:equal)
    savefig(p, "$(fname).svg")
end


# Calculate the matrix representation of the density operator given a basis
# Each λ and μ is a BitVector of length 2Nₒ where Nₒ is the number of orbital on each level. Functions basiscombine and basissplit may be useful.

function bilayer_density_matrix_element!(mat::SparseMatrixCSC{ComplexF64, Int64},  i::Int, j::Int, λ::BitVector,μ::BitVector, single_particle_function::Vector{ComplexF64})
    Nₒ = length(λ) ÷ 2 # Number of orbitals on each level.
    #print("$(Nₒ)\t\t")
    # Easiest way is to conver the bilayer basis into a single level (of length 2Nₒ) and use the single-layer code.
    check_difference = λ .⊻ μ
    count_difference = count(check_difference)
    #display(check_difference)
    #for thing in check_difference
    #    print("$(thing*1) ")
    #end
    #print(" | ")
    #println(count_difference)

    if count_difference == 2
        #print("Different indices: ")
        λ_a = bin2dex(check_difference.*λ)[1]
        μ_b  = bin2dex(check_difference.*μ)[1]
        a = count(λ[1:λ_a])
        b = count(μ[1:μ_b])
        #println("$(λ_a)\t$(μ_b)")
        if abs(λ_a-μ_b)<Nₒ # The two different elements must be in the same layer.    
            term = (-1)^(a+b) * conj.(single_particle_function[λ_a+1]) .* single_particle_function[μ_b+1]
            #println()
            # The +1 above is because Julia starts counting from 1. E.g. single_particle_function[1] == ϕ_0
            mat[i,j] += term
            mat[j,i] += conj(term)
        end
    elseif count_difference == 0
        #println(bin2dex(λ))
        term = sum([abs2.(single_particle_function[m+1]) for m in bin2dex(λ)])
        mat[i,j] += term
        #println(term)
    end   
end

function bilayer_density_element!(Nₒ::Int, den::Matrix{Float64}, coef::Number, λ::BitVector,μ::BitVector, single_particle_function::Vector{Matrix{ComplexF64}})
    #Nₒ = length(λ[1]) ÷ 2 # Number of orbitals on each level.
    check_difference = λ.⊻ μ
    count_difference = count(check_difference)
    if count_difference == 2
        λ_a = bin2dex(check_difference.*λ)[1]
        μ_b  = bin2dex(check_difference.*μ)[1]
        a = count(λ[1:λ_a])
        b = count(μ[1:μ_b])
        #print("$(λ_a)\t$(μ_b)\t$a\t$b\t")
        if abs(λ_a-μ_b)<Nₒ # The two different elements must be in the same layer.    
            #print("Updating density.")
            den .+= real.(coef * (-1)^(a+b) * conj.(single_particle_function[λ_a+1]) .* single_particle_function[μ_b+1])
        end
    elseif count_difference == 0
        #println(bin2dex(Lam))
       # print("Updating density.")
        den .+= real.(coef*sum([abs2.(single_particle_function[m+1]) for m in bin2dex(λ)]))
    end    
    #println()
end

function bilayer_density_matrix(basis_list::Vector{BitVector}, single_particle_function::Vector{ComplexF64})
    dim = length(basis_list)
    mat = spzeros(ComplexF64, (dim,dim))
    for i in 1:dim
        for j in i:dim
            print("\r$i\t$j\t")
            bilayer_density_matrix_element!(mat, i,j, basis_list[i], basis_list[j], single_particle_function)
        end
    end
    return mat
end

function bilayer_density_matrix(basis_list::Vector{BitVector}, ϵ::Number, z_0::ComplexF64)
    No = length(basis_list[1]) ÷ 2
    ϕ = single_particle_state_disk # Alias. Use ϕ(z, m ,Δ)
    single_particle_function= [ϕ.(z_0, m-No*(m>=No), ϵ*(m>=No)) for m in 0:2No]
    mat = bilayer_density_matrix(basis_list, single_particle_function)
    return mat
end

export Abstractbilayer_state, single_particle_state_disk,bilayer_state, 
        bilayer_state_mutable, disk_density, bilayerprintwf, 
        bilayerreadwf,display,bilayer_density_matrix,basissplit, 
        basiscombine, bilayer_density_element!
end # End of module