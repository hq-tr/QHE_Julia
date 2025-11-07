module IdealBand

include("Misc.jl")
using .MiscRoutine

using LinearAlgebra
using SparseArrays
using Combinatorics

abstract type IdealBandModel end

# Pauli Matrices
const σ₀ = [1. 0.; 0. 1.];
const σ₁ = [0. 1.; 1. 0.];
const σ₂ = [0. -1im ; 1im 0.];
const σ₃ = [1 0; 0 -1];

# A list of magic angles in TBG for easy reference
const MAGIC_ANGLES = [1.05,0.495,0.35]
const DEFAULT_HOPPINGS = [0.89,0.216]

struct LandauLevel <: IdealBandModel
end

struct IdealcTBG <: IdealBandModel
	twist_angle::Float64 # twist angle in degree
	w_0::Float64
	w_1::Float64
	IdealcTBG(twist_angle::Number) = new(float(twistangle),DEFAULT_HOPPINGS[1],DEFAULT_HOPPINGS[2])
	IdealcTBG(twist_angle::Symbol,w_0::Float64,w_1::Float64) = begin
		if twist_angle == :FIRST
			new(MAGIC_ANGLES[1],w_0,w_1)
		elseif twist_angle == :SECOND
			new(MAGIC_ANGLES[2],w_0,w_1)
		elseif twist_angle == :THIRD
			new(MAGIC_ANGLES[3],w_0,w_1)
		else
			error("Keyword not recognized.")
		end
	end
	IdealcTBG(twist_angle::Symbol) = new(twist_angle,DEFAULT_HOPPINGS[1],DEFAULT_HOPPINGS[2])
end

function reciprocal_vectors(model::IdealcTBG)
	θ = model.twist_angle*2π/360 # convert twist angle to radian
	a = 2*sin(θ/2)*sqrt(2π/sin(pi/3))

	G1 = 4π/a*sin(θ/2)*Vector{ComplexF64}([ 1/sqrt(3), 1]) 
	G2 = 4π/a*sin(θ/2)*Vector{ComplexF64}([-1/sqrt(3), 1])
	return G1, G2
end

function reciprocal_vectors(model::LandauLevel)
	return Vector{ComplexF64}([1.0,0.0]),Vector{ComplexF64}([0.0,1.0])
end

# Calculate single-particle and two-particle matrix elments
# of one-body and two-body potentials

function potential_pin_element(model::LandauLevel,k1x::Float64,k1y::Float64,k2x::Float64,k2y::Float64,x0::Float64,y0::Float64,range_bx::Int=2,range_by::Int=2)
	#b1,b2 = reciprocal_vectors(model)
	term = 0.0 + 0.0im
	for bx in -range_bx:range_bx, by in -range_by:range_by # here bx and by are coefficient of b in b₁, b₂ basis
		term += exp(1im*((k1x+bx-k2x)*x0 + (k1y+by-k2y)*y0)) * exp(-((k1x+bx-k2x)^2+(k1y+by-k2y)^2)/4) * exp(0.5im*(k1x*k2y-k1y*k2x)) * exp(0.5im*(bx*(k1y+k2y) - by*(k1x+k2x)))
	end
	return term
end


η(bx::Int,by::Int) = (bx%2==0) && (by%2==0) ? 1 : -1 
function η(bx::Number,by::Number)
	try 
		return η(Int(bx),Int(by))
	catch InexactError
		return 0
	end
end

function two_body_element(model::LandauLevel,v_q::Function,k1x::Float64,k1y::Float64,k2x::Float64,k2y::Float64,k3x::Float64,k3y::Float64,k4x::Float64,k4y::Float64,range_bx::Int=2,range_by::Int=2)
	#b1,b2 = reciprocal_vectors(model)
	δbx = k1x+k2x-k3x-k4x
	δby = k1y+k2y-k3y-k4y
	if !isinteger(δbx) || !(isinteger(δby))
		return 0.0
	else
		term = 0.0 + 0.0im
		for bx in -range_bx:range_bx, by in -range_by:range_by # here bx and by are coefficient of b in b₁, b₂ basis
			term += v_q(k1x-k4x-bx,k1y-k4y-by) * exp(-2π*((k1x-k4x-bx)^2+(k1y-k4y-by)^2)/2) * exp(0.5im*2π*((k1x-k4x-bx)*(k4y-k3y)-(k1y-k4y-by)*(k4x-k3x))) * η(bx,by) * η(Int(δbx),Int(δby)) * exp(0.5*2π*im*(bx*k1y-by*k1x)) * exp(-0.5im*2π*((bx-δbx)*k2y - (by-δby)*k2x))
		end
		return term
	end
end

# Calculate the two-body interaction matrix of many-body state

function update_element!(model::IdealBandModel,H_matrix::SparseMatrixCSC{ComplexF64}, N_o::Int64, 
			i::Int64, j::Int64, basis1::BitVector, basis2::BitVector, Nx::Int64,Ny::Int64,v_q::Function)
	if i == j
		basis = findall(basis1)
		for (k1,k2) in combinations(basis,2)
			k1x,k1y = get_k_vector(k1,Nx,Ny)
			k2x,k2y = get_k_vector(k2,Nx,Ny)
			term = two_body_element(model,v_q,k1x,k1y,k2x,k2y,k1x,k1y,k2x,k2y)
			if abs(term) > 1e-10
				H_matrix[i,i] += abs(term)
			end
		end
	else
		b = basis1 .⊻ basis2
		
		if sum(b) == 4
			m1m2 = findall(basis1 .& b)
			m3m4 = findall(basis2 .& b)
			k1x,k1y = get_k_vector(m1m2[1],Nx,Ny)
			k2x,k2y = get_k_vector(m1m2[2],Nx,Ny)
			k3x,k3y = get_k_vector(m3m4[1],Nx,Ny)
			k4x,k4y = get_k_vector(m3m4[2],Nx,Ny)
			c = count(basis1[m1m2[1]:m1m2[2]])
			d = count(basis2[m3m4[1]:m3m4[2]])
			term = (-1)^(c+d) * two_body_element(model,v_q,k1x,k1y,k2x,k2y,k3x,k3y,k4x,k4y)
			if abs(term) > 1e-10
				H_matrix[i,j] += term
				H_matrix[j,i] += conj(term)
			end
		end
	end
	return
end

# This is the matrix for two-body interaction given a basis
function two_body(model::IdealBandModel,N_o::Int64, Nx::Int,Ny::Int,basis::Vector{BitVector},v_q::Function)
	dim = length(basis)
	println("The dimension is $(dim)")
	H_matrix = spzeros(ComplexF64,dim, dim)
	for i in 1:dim
		#print("\rRow $(i)\t\t")
		for j in i:dim
			update_element!(model,H_matrix, N_o, i, j, basis[i], basis[j],Nx,Ny,v_q)
		end
	end
#	display(H_matrix)
	return H_matrix
end

export IdealBandModel, LandauLevel, IdealcTBG, reciprocal_vectors, potential_pin_element, two_body
end # End module
