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

import Base.display

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

function split(vec::Abstractbilayer_state)
    state1 = disk_normalize(FQH_state(map(x->x[1],vec.basis), vec.coef))
    state2 = disk_normalize(FQH_state(map(x->x[2],vec.basis), vec.coef),0.5)
    return state1, state2
end

function printwf(vec::Abstractbilayer_state;fname="")
    combined_basis = map(x->vcat(x[1], x[2]), vec.basis)
    state = FQH_state(combined_basis, vec.coef)
    D = dim(state)
    if length(fname) == 0
        println(D)
        for i in 1:D
            println(prod(string.((Int.(state.basis[i])))))
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

single_particle_state_disk(z::Number,m::Integer,Δ::Number) = z.^(m+Δ) * exp.(-abs.(z)^2) * sqrt(2^m / (2π)) / sqrt(gamma(m+Δ))

single_particle_state_disk(z::Vector{T} where T <: Number,m::Integer, Δ::Number) = z.^(m+Δ) * exp.(-abs.(z)^2) * sqrt(2^m / (2π)) / sqrt(gamma(m+Δ))

function disk_density(vec::Abstractbilayer_state)
    state1, state2 = split(vec)
    # Prepare the plot area
    No = length(state1.basis[1])
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

export Abstractbilayer_state, bilayer_state, bilayer_state_mutable, disk_density, printwf, display
end # End of module