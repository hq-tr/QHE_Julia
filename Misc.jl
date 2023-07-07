module MiscRoutine

# Basis format conversions
bin2dex(config::BitVector) = [i-1 for i in 1:length(config) if config[i]]

dex2bin(state::Vector{Int}, N_orb::Int) = BitVector([i-1 in state for i in 1:N_orb]) 

# Normalization coefficient on the sphere
sphere_coef(S,m) =   sqfactorial(S-m)/sqfactorial(S+m+1, 2S+1)

# Miscellaneous functions for LLL physics

findLz(root::BitVector) = sum(root .* collect(0:1:length(root)-1))

findLzsphere(root::BitVector, S::Float64) = sum(root .* collect(-S:1:S))

function findLzsphere(root::BitVector)
	S = (length(BitVector)-1.)/2.
	return findLzsphere(root)
end

sqfactorial(N) = prod(map(sqrt, 1:N)) 
# square root of N!, avoid overflow for large N and more efficient than sqrt(factorial(big(N)))
# (overflow starts at N=21)

sqfactorial(n,N) = prod(map(sqrt, n:N))

export bin2dex, sqfactorial, dex2bin, findLz, findLzsphere, sphere_coef
end