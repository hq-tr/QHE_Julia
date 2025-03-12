# ------------------ MODULE FOR CALCULATING THE ELECTRON DENSITY
module ParticleDensity

include("Misc.jl")
using .MiscRoutine

using SparseArrays
using HypergeometricFunctions

# LLL single-particle states
single_particle_state_disk(z::Number,m::Integer) = z.^m * exp.(-abs.(z)^2) * sqrt(2^m / (2π)) / sqfactorial(m)

single_particle_state_disk(z::Vector{T} where T <: Number,m::Integer) = z.^m * exp.(-abs.(z)^2) * sqrt(2^m / (2π)) / sqfactorial(m)

single_particle_state_sphere(θ::Number, φ::Number, S::Number, m::Number) = cos(θ/2)^(S+m) * sin(θ/2)^(S-m) * exp(m*im*φ) / sphere_coef(S,m)

single_particle_state_sphere(θ::Array{T} where T<: Number, φ::Array{T} where T<: Number, S::Number, m::Number) = cos.(θ./2).^(S.+m) .* sin.(θ/2).^(S.-m) .* exp.(m*im.*φ) / sphere_coef(S,m)

# Higher LL single-particle states
# Same function names but taking one additional argument (n = LL index)

# general laguerre
genlaguerre(n::Integer,α::Integer,x::Number) = binomial(n+α,n) * HypergeometricFunctions.M(-n, α+1,x)
genlaguerre(n::Integer,α::Integer,x::Vector{T} where T<:Number) = collect(map(xx -> genlaguerre(n,α,xx) for xx in x))

single_particle_state_disk(z::Number,m::Integer,n::Integer) = z.^m * exp.(-abs.(z)^2) * (-1)^n / (sqrt(2π*2^m) * sqfactorial(n+1,n+m)) * genlaguerre(n,m,abs2(z)/2)
single_particle_state_disk(z::Vector{T} where T <: Number,m::Integer, n::Integer) = z.^m * exp.(-abs.(z)^2) / (sqrt(2π*2^m) * sqfactorial(n+1,n+m)) * genlaguerre(n,m,abs2.(z)./2)

function density_element_disk(Lam::BitVector,Mu::BitVector, x::Real,y::Real)
    z = 0.5(x+y*im)
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        return (-1)^(a+b) * conj(single_particle_state_disk(z, Lam_a)) * single_particle_state_disk(z,Mu_b)
    elseif count_difference == 0
        #println("Same partition")
        #println(bin2dex(Lam))
        return sum([abs(single_particle_state_disk(z,m))^2 for m in bin2dex(Lam)])
    else
        return 0
    end    
end

function density_element_sphere(Lam::BitVector,Mu::BitVector, θ::Real, φ::Real)
    No = length(Lam)
    S  = (No-1)/2.
    println(S)
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        return (-1)^(a+b) * conj(single_particle_state_sphere(θ, φ, S, S-Lam_a)) * single_particle_state_sphere(θ,φ,S,S-Mu_b)
    elseif count_difference == 0
        #println("Same partition")
        #println(bin2dex(Lam))
        return sum([abs(single_particle_state_sphere(θ,φ,S,S-m))^2 for m in bin2dex(Lam)])
    else
        return 0
    end    
end

function density_element_gen(Lam::BitVector,Mu::BitVector, single_particle_function::Vector{T} where T <: Number)
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        return (-1)^(a+b) * conj(single_particle_function[Lam_a+1]) * single_particle_function[Mu_b+1]
    elseif count_difference == 0
        #println("Same partition")
        #println(bin2dex(Lam))
        return sum([abs2(single_particle_function[m+1]) for m in bin2dex(Lam)])
    else
        return 0
    end    
end

function density_element_gen(Lam::BitVector,Mu::BitVector, single_particle_function::Vector{Matrix{ComplexF64}})
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        return (-1)^(a+b) .* conj.(single_particle_function[Lam_a+1]) .* single_particle_function[Mu_b+1]
    elseif count_difference == 0
        #println("Same partition")
        #println(bin2dex(Lam))
        return sum([abs2.(single_particle_function[m+1]) for m in bin2dex(Lam)])
    else
        return zeros(size(single_particle_function[1]))
    end    
end

function density_element_gen!(den::Matrix{Float64}, coef::Number, Lam::BitVector,Mu::BitVector, single_particle_function::Vector{Matrix{ComplexF64}})
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        den .+= real.(coef * (-1)^(a+b) * conj.(single_particle_function[Lam_a+1]) .* single_particle_function[Mu_b+1])
    elseif count_difference == 0
        #println(bin2dex(Lam))
        den .+= real.(coef*sum([abs2.(single_particle_function[m+1]) for m in bin2dex(Lam)]))
    end    
end

function density_element_gen!(den::Vector{Float64}, coef::Number, Lam::BitVector,Mu::BitVector, single_particle_function::Vector{Vector{ComplexF64}})
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        den .+= real.(coef * (-1)^(a+b) * conj.(single_particle_function[Lam_a+1]) .* single_particle_function[Mu_b+1])
    elseif count_difference == 0
        #println(bin2dex(Lam))
        den .+= real.(coef*sum([abs2.(single_particle_function[m+1]) for m in bin2dex(Lam)]))
    end    
end

function density_matrix_element!(mat::SparseMatrixCSC{ComplexF64, Int64},  i::Int, j::Int, Lam::BitVector,Mu::BitVector, single_particle_function::Vector{ComplexF64})
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        term = (-1)^(a+b) * conj.(single_particle_function[Lam_a+1]) .* single_particle_function[Mu_b+1]
        mat[i,j] += term
        mat[j,i] += conj(term)
    elseif count_difference == 0
        #println(bin2dex(Lam))
        mat[i,j] += sum([abs2.(single_particle_function[m+1]) for m in bin2dex(Lam)])
    end   
end

function density_matrix(basis_list::Vector{BitVector}, single_particle_function::Vector{ComplexF64})
    dim = length(basis_list)
    mat = spzeros(ComplexF64, (dim,dim))
    for i in 1:dim
        for j in i:dim
            density_matrix_element!(mat, i,j, basis_list[i], basis_list[j], single_particle_function)
        end
    end
    return mat
end

export density_element_disk, density_element_sphere, density_element_gen, single_particle_state_disk, single_particle_state_sphere, density_element_gen!, density_matrix


end
