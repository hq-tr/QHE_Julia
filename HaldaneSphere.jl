include("FQH_state_v2.jl")
module HaldaneSphere
include("FQH_state_v2.jl")
include("Misc.jl")

using .FQH_states
using .MiscRoutine

# function one_particle_state(θ::Float64,ϕ::Float64,S2::Int;fname="") # S2 = 2S where S is the monopole strength
# 	u = cos(θ/2) * exp(-0.5im * ϕ)
# 	v = sin(θ/2) * exp(0.5im * ϕ)
# 	basis = BitVector[]
# 	coef  = ComplexF64[]
# 	for i in 0:(S2)
# 		push!(basis, BitVector([j==i for j in 0:(S2)]))
# 		push!(coef, u^(S2-i) * v^i / sphere_coef(S2/2.0, S2/2.0 -i))
# 	end
# 	state = wfnormalize(FQH_state(basis, coef))
# 	if length(fname)>0
# 		printwf(state;fname=fname)
# 	end
# 	return state
# end

function one_particle_state_coef(θ::Float64,ϕ::Float64,S2::Int)
	u = cos(θ/2) * exp(-0.5im * ϕ)
	v = sin(θ/2) * exp(0.5im * ϕ)
	return map(i-> u^(S2-i) * v^i / sphere_coef(S2/2.0, S2/2.0 -i), 0:S2)
end

function one_particle_state_coef(m::Int,θ::Float64,ϕ::Float64,S2::Int)
	if m<0 || m>S2
		throw(DomainError(m, "m must be between 0 and S2 (inclusive)"))
	elseif m==0
		ret = one_particle_state_coef(θ,ϕ,S2)
	else
		u = cos(θ/2) * exp(-0.5im * ϕ)
		v = sin(θ/2) * exp(0.5im * ϕ)
		ret = one_particle_state_coef(m-1,θ,ϕ,S2)
	end
		
	return map(i-> u^(S2-i) * v^i / sphere_coef(S2/2.0, S2/2.0 -i), 0:S2)
end

function one_particle_state(θ::Float64,ϕ::Float64,S2::Int;fname="")
	basis = [BitVector([j==i for j in 0:(S2)]) for i in 0:(S2)]
	coefs = one_particle_state_coef(θ,ϕ,S2)
	state = wfnormalize(FQH_state(basis, coefs))
	if length(fname)>0
		printwf(state;fname=fname)
	end
	return state
end


function split_particle_state(θ::Float64,ϕ::Float64,S2::Int;fname="") # This one is wrong lol
	u = cos(θ/2) * exp(-0.5im * ϕ)
	v = sin(θ/2) * exp(0.5im * ϕ)
	uu = cos(θ/2) * exp(-0.5im * ϕ)
	vv = sin(θ/2) * exp(0.5im * ϕ)
	basis = BitVector[]
	coef  = ComplexF64[]
	for i in 0:(S2)
		push!(basis, BitVector([j==i for j in 0:(S2)]))
		push!(coef, ( u^(S2-i) * v^i + uu^(S2-i) * vv^i ) / sphere_coef(S2/2.0, S2/2.0 -i))
	end
	state = wfnormalize(FQH_state(basis, coef))
	if length(fname) > 0
		printwf(state;fname=fname)
	end
	return state
end

function split_particle_state_coef(θ::Float64,ϕ::Float64,S2::Int)
	u = cos(θ/2) * exp(-0.5im * ϕ)
	v = sin(θ/2) * exp(0.5im * ϕ)
	uu = cos(θ/2) * exp(-0.5im * ϕ)
	vv = sin(θ/2) * exp(0.5im * ϕ)
	return map(i->( u^(S2-i) * v^i + uu^(S2-i) * vv^i ) / sphere_coef(S2/2.0, S2/2.0 -i), 0:S2)
end

export one_particle_state, split_particle_state, one_particle_state_coef,split_particle_state
end




function main()
	println("Creating a coherent state on the sphere.")
	println("Input θ and ϕ (as a multiple of π):")
	θ,ϕ = map(x->parse(Float64,x) * π, split(readline()))

	println("Input N_orb:")
	No = parse(Int, readline())

	@time HaldaneSphere.one_particle_state(θ,ϕ,No-1;fname="test_state")
end

main()
