include("FQH_state_v2.jl")
using LinearAlgebra
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

function nCr(n,r)
	return r>0 ? prod(k->(n-r+k)/k,1:r) : 1
end

function sqnCr(n,r)
	return r>0 ? prod(k->√((n+r+k)/k), 1:r) : 1
end

function sqfactorial(n)
	return n>0 ? prod(x->√(x), 1:n) : 1
end

# The function below calculates the Wigner d-matrix d_{m1,m2} = <j,m_1|e^{iθL_y}|j,m_2>
# Courtersy of DeepSeek (with some modifications)
function wignerd(theta, j, m1, m2)
	#println("Matrix element d_($m1,$m2)")
    # Check if inputs are valid
    if !(j >= 0 && isinteger(2j))
        throw(ArgumentError("j must be integer or half-integer ≥ 0"))
    end
    if !(isinteger(m1) && isinteger(m2))
        throw(ArgumentError("m1 and m2 must be integer or half-integer"))
    end
    if abs(m1) > j || abs(m2) > j
        throw(ArgumentError("|m1|, |m2| must be ≤ j"))
    end
    
    # Convert to rational to avoid floating point issues in factorial
    j_r = rationalize(j)
    m1_r = rationalize(m1)
    m2_r = rationalize(m2)
    
    # Use symmetry properties to reduce computation
    if abs(m1) < abs(m2)
        # Use symmetry: d^{j}_{m1,m2}(θ) = (-1)^{m1-m2} d^{j}_{m2,m1}(θ)
        return (-1)^(m1-m2) * wignerd(theta, j, m2, m1)
    end
    
    if theta < 0
        # Use symmetry: d^{j}_{m1,m2}(-θ) = d^{j}_{m2,m1}(θ)  
        return wignerd(-theta, j, m2, m1)
    end
    
    # Special case: θ = 0
    if theta == 0
        return m1 == m2 ? 1.0 : 0.0
    end
    
    # Special case: θ = π
    if theta ≈ π
        return (-1)^(j-m2) * (m1 == -m2 ? 1.0 : 0.0)
    end
    
    # Precompute factorials for efficiency
    function log_factorial(x)
        x < 0 && return -Inf
        x == 0 && return 0.0
        return sum(log(i) for i in 1:x)
    end
    
    # Initialize sum
    result = 0.0
    c = cos(-theta/2) # Due to different sign conventions, the theta in the formula is actually -theta in my codes
    s = sin(-theta/2)
    
    # Determine summation bounds for k
    k_min = max(0, m1 - m2, 0)  # k ≥ 0 and k+m2-m1 ≥ 0
    k_max = min(j + m1, j - m2, j - m1 + m2 - k_min + 100)  # Upper bounds from factorials
    
    #print("Range of sum over k = ")
    #println((k_min,k_max))
    for k in k_min:k_max
        # Check factorial arguments are non-negative
        if (j + m1 - k < 0) || (j - m2 - k < 0) || (k + m2 - m1 < 0)
            continue
        end
        
        # Calculate term using logarithms to avoid overflow
        log_term = 0.5 * (log_factorial(j + m1) + log_factorial(j - m1) + 
                          log_factorial(j + m2) + log_factorial(j - m2))
        
        log_term -= (log_factorial(j + m1 - k) + log_factorial(j - m2 - k) + 
                     log_factorial(k) + log_factorial(k + m2 - m1))
        
        # Add sign and trigonometric factors
        term = exp(log_term) * (-1)^k
        
        # Trigonometric factors
        cos_power = 2j + m1 - m2 - 2k
        sin_power = 2k + m2 - m1
        
        term *= c^cos_power * s^sin_power
        
        result += term
    end
    
    return result
end

function one_particle_state_coef(θ::Float64,ϕ::Float64,S2::Int; normalize=false)
	u = cos(θ/2) * exp(-0.5im * ϕ)
	v = sin(θ/2) * exp(0.5im * ϕ)
	ret = map(i-> u^(S2-i) * v^i / sphere_coef(S2/2.0, S2/2.0 -i), 0:S2)
	if normalize
		ret./= ret' * ret
	end
	return ret
end

function one_particle_state_coef(m::Int,θ::Float64,ϕ::Float64,S2::Int;normalize=false)
	S = S2/2.
	ret = [wignerd(θ,S,k,-S+m) for k in -S:S]
	if ϕ ≈ π
		ret .*= -1
	elseif !(ϕ ≈ 0)
		ret = ret.*exp.(1im*ϕ .* (-S:S))
	end
	if normalize
		ret ./= ret' * ret
	end
	return ret
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

function one_particle_state(m::Int,θ::Float64,ϕ::Float64,S2::Int;fname="")
	basis = [BitVector([j==i for j in 0:(S2)]) for i in 0:(S2)]
	coefs = one_particle_state_coef(m,θ,ϕ,S2)
	state = wfnormalize(FQH_state(basis, coefs))
	display(state.coef)
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

export one_particle_state, split_particle_state, one_particle_state_coef,wignerd
end




function main()
	println("Creating a coherent state on the sphere.")
	println("Input θ and ϕ (as a multiple of π):")
	θ,ϕ = map(x->parse(Float64,x) * π, split(readline()))

	println("Input N_orb:")
	No = parse(Int, readline())

	@time state = HaldaneSphere.one_particle_state(1,θ,ϕ,No-1;fname="test_state")
	println("-----")
	c = HaldaneSphere.one_particle_state(θ,ϕ,No-1)
	display(c.coef)
	println("-----")
	display(abs(state⋅c))
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end
