module HaldaneSphere
include("FQH_state_v2.jl")
include("Misc.jl")

using .FQH_states
using .MiscRoutine

function one_particle_state(θ::Float64,ϕ::Float64,S2::Int)
	u = cos(θ/2) * exp(-0.5im * ϕ)
	v = sin(θ/2) * exp(0.5im * ϕ)
	basis = BitVector[]
	coef  = ComplexF64[]
	for i in 0:(S2)
		push!(basis, BitVector([j==i for j in 0:(S2)]))
		push!(coef, u^(S2-i) * v^i / sphere_coef(S2/2.0, S2/2.0 -i))
	end
	state = wfnormalize(FQH_state(basis, coef))
	printwf(state;fname="state")
	return state
end

export one_particle_state
end

#=
function main()
	println("Creating a coherent state on the sphere.")
	println("Input θ and ϕ (as a multiple of π):")
	θ,ϕ = map(x->parse(Float64,x) * π, split(readline()))

	println("Input N_orb:")
	No = parse(Float64, readline())

	@time one_particle_state(θ,ϕ,No-1)
end

main()
=#