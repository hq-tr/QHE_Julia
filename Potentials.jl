module Potentials
include("Misc.jl")
include("FQH_state_v2.jl")
include("HaldaneSphere.jl")
include("Density.jl")
using .FQH_states
using .MiscRoutine
using .HaldaneSphere
using .ParticleDensity

using LinearAlgebra
using SparseArrays
using Arpack


function gen_onebody_element!(mat::SparseMatrixCSC{ComplexF64, Int64}, i::Int, j::Int, Lam::BitVector,Mu::BitVector, C::Matrix{T} where T<:Number, height::Float64)
    if i==j
        mat[i,j] += height * sum([C[m+1,m+1] for m in bin2dex(Lam)])
    else
        #check_difference = Lam .⊻ Mu
        #count_difference = count(check_difference)
        diff_index_Lam   = findall(Lam .& .!Mu) # all indices where Lam is 1 and Mu is 0. Indexing starts from 1
        #if count_difference == 2
        if length(diff_index_Lam) == 1
            #Lam_a = bin2dex(check_difference.*Lam)[1]
            #Mu_b  = bin2dex(check_difference.*Mu)[1]
            Lam_a = diff_index_Lam[1]
            Mu_b  = findall(Mu .& .!Lam)[1]
            a = count(Lam[1:(Lam_a-1)])
            b = count(Mu[1:(Mu_b-1)])
            term = height * (-1)^(a+b) * C[Lam_a, Mu_b]
            mat[i,j] += term
            mat[j,i] += conj(term)
        #elseif count_difference == 0
        #elseif length(diff_index_Lam) == 0
            #println(bin2dex(Lam))
            #mat[i,j] += height * sum([C[m+1,m+1] for m in bin2dex(Lam)])
        end   
    end 
end

function gen_onebody_element_diagonal!(mat::SparseMatrixCSC{ComplexF64, Int64}, i::Int, Lam::BitVector, C::Vector{T} where T<:Number, height::Float64)
    mat[i,i] += height * sum([C[m+1] for m in bin2dex(Lam)])  
end

function gen_onebody_element_diagonal!(mat::SparseMatrixCSC{ComplexF64, Int64}, i::Int, Lam::BitVector, C::Matrix{T} where T<:Number, height::Float64)
    mat[i,i] += height * sum([C[m+1,m+1] for m in bin2dex(Lam)])  
end

function gen_onebody_matrix(basis_list::Vector{BitVector}, C::Matrix{T} where T<:Number,height=1.,shift=0.)
    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function gen_onebody_groundstate(basis_list::Vector{BitVector}, C::Matrix{T} where T<:Number,shift=0.;fname="")
    mat = gen_onebody_matrix(basis_list, C, 1., shift)
    E, V = eigs(mat, nev=5, sigma=0)
    println("Energy eigenvalues:")
    display(E)
    gs = FQH_state(basis_list, V[:,1])
    if !isempty(fname) printwf(gs, fname=fname) end
    return gs    
end

# This function updates the elements for one pin with a given pos and height
function diracdelta_element!(mat::SparseMatrixCSC{ComplexF64, Int64}, i::Int, j::Int, Lam::BitVector,Mu::BitVector, pos::Number, height::Number)
    R = abs(pos)
    θ = angle(pos) + π
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        term = height * (-1)^(a+b) * (-1)^(Mu_b-Lam_a) * π * R^(Lam_a+Mu_b) * exp(-R^2/2) * exp(im*(Mu_b-Lam_a)*θ)/((√2)^(Lam_a+Mu_b)*sqfactorial(Lam_a)*sqfactorial(Mu_b))
        mat[i,j] += term
        if i!=j mat[j,i] += conj(term) end
    elseif count_difference == 0
        #println(bin2dex(Lam))
        for m in bin2dex(Lam)
            mat[i,j] += height *  π*R^(2m) * exp(-R^2/2)/(2^m*factorial(big(m)))
        end
    end    
end

# This function updates the elements for multiple pins with a different positions and heights
function diracdelta_element!(mat::SparseMatrixCSC{ComplexF64, Int64}, i::Int, j::Int, Lam::BitVector,Mu::BitVector, 
            pos::Vector{T} where T<:Number, heights::Vector{T} where T<:Number)
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        for (po,height) in zip(pos,heights)
            R = abs(po)
            θ = angle(po) + π
            term = height * (-1)^(a+b) * (-1)^(Mu_b-Lam_a) * π * R^(Lam_a+Mu_b) * exp(-R^2/2) * exp(im*(Mu_b-Lam_a)*θ)/((√2)^(Lam_a+Mu_b)*sqfactorial(Lam_a)*sqfactorial(Mu_b))
            
            mat[i,j] += term
            if i!=j mat[j,i] += conj(term) end
        end
    elseif count_difference == 0
        #println(bin2dex(Lam))
        for (po,height) in zip(pos,heights)
            R = abs(po)
            θ = angle(po) + π
            for m in bin2dex(Lam)
                mat[i,j] += height *  π*R^(2m) * exp(-R^2/2)/(2^m*factorial(big(m)))
            end
        end
    end    
end

# this function constructs the matrix for one pin with a given pos and height
function diracdelta_matrix(basis_list::Vector{BitVector}, pos::Number,height=1.,shift=0.)
    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            diracdelta_element!(mat, i, j, basis_list[i], basis_list[j], pos, height)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

# this function constructs the matrix for multiple pins with positions given by pos, all of the same height
function diracdelta_matrix(basis_list::Vector{BitVector}, pos::Vector{T} where T<:Number,height::Float64=1.,shift=0.)
    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            for po in pos
                diracdelta_element!(mat, i, j, basis_list[i], basis_list[j], po,height)
            end
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 


# this function constructs the matrix for multiple pins with different positions and heights
function diracdelta_matrix(basis_list::Vector{BitVector}, pos::Vector{T} where T<:Number,heights::Vector{T} where T<:Number,shift=0.)
    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            diracdelta_element!(mat, i, j, basis_list[i], basis_list[j], pos,heights)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function diracdelta_groundstate(basis_list::Vector{BitVector}, pos::Number,shift=0.;fname="")
    mat = diracdelta_matrix(basis_list, pos, shift)
    E, V = eigs(mat, nev=5, sigma=0)
    println("Energy eigenvalues:")
    display(E)
    gs = FQH_state(basis_list, V[:,1])
    if !isempty(fname) printwf(gs, fname=fname) end
    return gs    
end

function diracdelta_groundstate(basis_list::Vector{BitVector}, pos::Vector{T} where T<:Number, shift=0.;fname="groundstate")
    # More than one trap
    dim = length(basis_list)
    mat = spzeros(Complex{Float64}, (dim,dim))
    for p in pos
        mat += diracdelta_matrix(basis_list, p)
    end
    if shift!=0
        mat += shift * sparse(I,dim,dim)
    end
    E, V = eigs(mat, nev=5, sigma=0)
    println("Energy eigenvalues:")
    display(E)
    gs = FQH_state(basis_list, V[:,1])
    if !isempty(fname) printwf(gs; fname=fname) end
    return gs    
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


function sphere_bump_matrix(basis_list::Vector{BitVector},θ::Real, ϕ::Real, height=1.0,shift=0.0;verbose=false)
    No = length(basis_list[1])
    coef = one_particle_state(θ,ϕ,No-1).coef
    C  = coef * coef'

    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        if verbose && ((i-1)%(dim÷100) == 0)
            print("\rProgress: $(i÷(dim÷100))%\t\t")
        end
        for j in i:dim
            gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

# This is for multiple pins of the same height
function sphere_bump_matrix(basis_list::Vector{BitVector},θ::Vector{Float64}, ϕ::Vector{Float64}, height=1.0,shift=0.0;verbose=false)
    No = length(basis_list[1])

    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))

    C = sum(k->begin 
        coef = one_particle_state(θ[k],ϕ[k],No-1).coef
        coef * coef'
        end,1:length(θ))
    for i in 1:dim
        if verbose && ((i-1)%(dim÷100) == 0)
            print("\rProgress: $(i÷(dim÷100))%\t\t")
        end
        for j in i:dim
            gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
        end
    end


    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

# The following function only works if the bump is at the north or south pole
function sphere_wide_bump_matrix(basis_list::Vector{BitVector},k::Int=2, height::Float64=1.0,shift::Float64=0.0,location::Symbol=:north)
    No = length(basis_list[1])
    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    
    if location == :north
        C = vcat(ones(Float64,k), zeros(Float64,No-k))
    elseif location == :south
        C = vcat(zeros(Float64,No-k),ones(Float64,k))
    else
        println("WARNING: 'location' argument must be either :north or :south")
        println("Otherwise, an empty matrix will be returned.")
        return mat
    end
    for i in 1:dim
        gen_onebody_element_diagonal!(mat, i, basis_list[i], C, height)
    end
    if !iszero(shift) mat += shift * sparse(I, dim, dim) end
    return mat
end 


# This function calculate the matrix for one wide bump consisting of k orbital located at (θ,ϕ)
function sphere_wide_bump_matrix(basis_list::Vector{BitVector},k::Int, θ::Real, ϕ::Real,height::Float64=1.0,shift::Float64=0.0;verbose=false)
    No = length(basis_list[1])
    C = zeros(ComplexF64,(No,No)) # one-particle basis matrix
    for kk in 0:(k-1) # sum over orbitals 0,1,2,..,k-1
        coef = one_particle_state_coef(kk,θ,ϕ,No-1;normalize=true)
        C   += coef * coef'
    end

    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        if verbose && ((i-1)%(dim÷100) == 0)
            print("\rProgress: $(i÷(dim÷100))%\t\t")
        end
        for j in i:dim
            gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

# This function calculate the matrix for multiple wide bumps, each consisting of k orbital.
function sphere_wide_bump_matrix(basis_list::Vector{BitVector},k::Int, θ::Vector{T} where T<:Real, ϕ::Vector{T} where T<:Real,height::Float64=1.0,shift::Float64=0.0;verbose=false)
    No = length(basis_list[1])

    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))

    C = zeros(ComplexF64,(No,No)) # one-particle basis matrix
    for kk in 0:(k-1) # sum over orbitals 0,1,2,..,k-1
        C += sum(i->begin 
            coef = one_particle_state_coef(kk,θ[i],ϕ[i],No-1;normalize=true)
            coef * coef'
            end,1:length(θ))
    end

    for i in 1:dim
        if verbose && ((i-1)%(dim÷100) == 0)
            print("\rProgress: $(i÷(dim÷100))%\t\t")
        end
        for j in i:dim
            gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
        end
    end


    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function sphere_twinbump_matrix(basis_list::Vector{BitVector},θ::Float64, ϕ::Float64, height=1.0,shift=0.0)
    No = length(basis_list[1])
    coef = one_particle_state_coef(θ,ϕ,No-1) + one_particle_state_coef(θ,ϕ+π,No-1)
    C  = coef * coef'

    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function sphere_point_matrix(basis_list::Vector{BitVector},θ::Float64, ϕ::Float64, height=1.0,shift=0.0)
    No = length(basis_list[1])

    S = (No-1.0)/2.0
    sfunction = map(m->single_particle_state_sphere(π-θ,ϕ,S,m), -S:1:S)
    
    mat = height.*density_matrix(basis_list, sfunction)

    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

# One pin at (θ,ϕ) and one pin at (θ,ϕ+π)
function sphere_twinpoint_matrix(basis_list::Vector{BitVector},θ::Float64, ϕ::Float64, height=1.0,shift=0.0)
    No = length(basis_list[1])
    dim = length(basis_list)
    S = (No-1.0)/2.0

    sfunction = map(m->single_particle_state_sphere(π-θ,ϕ,S,m), -S:1:S)
    
    mat = spzeros(ComplexF64,(dim,dim))

    for i in 1:dim
        for j in i:dim
            sphere_twinpoint_element!(mat, i,j, basis_list[i], basis_list[j], sfunction)
        end
    end

    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function sphere_twinpoint_element!(mat::SparseMatrixCSC{ComplexF64, Int64},  
    i::Int, j::Int, Lam::BitVector,Mu::BitVector, 
    single_particle_function::Vector{ComplexF64})
    if i==j
        mat[i,j] += 2* sum([abs2.(single_particle_function[m+1]) for m in bin2dex(Lam)])
    else
        check_difference = Lam .⊻ Mu
        if count(check_difference) == 2
            Lam_a = bin2dex(check_difference.*Lam)[1]
            Mu_b  = bin2dex(check_difference.*Mu)[1]
            if (Lam_a + Mu_b) % 2 == 0
                a = count(Lam[1:Lam_a])
                b = count(Mu[1:Mu_b])
                term = 2 .* (-1)^(a+b) .* conj.(single_particle_function[Lam_a+1]) .* single_particle_function[Mu_b+1]
                mat[i,j] += term
                mat[j,i] += conj(term)
            end
        end
    end
end

export diracdelta_matrix, diracdelta_groundstate, diracdelta_element!, 
gen_onebody_matrix, gen_onebody_element!, gen_onebody_groundstate, 
sphere_bump_matrix, sphere_point_matrix, sphere_twinpoint_matrix,
sphere_twinbump_matrix,sphere_wide_bump_matrix

end