include("HilbertSpace.jl")
using Main.HilbertSpaceGenerator

include("FQH_state_v2.jl")
using Main.FQH_states

using BenchmarkTools
println("Input N_el: ")
Ne = parse(Int, readline())

println("Input N_orb: ")
No = parse(Int, readline())

@time begin
function isadmissible(partition::BitVector, k::Integer, r::Integer)
    check = true
    for i in 1:(length(partition)-r+1)
        if count(partition[i:(i+r-1)]) > k
            check=false
            break
        end
    end
    return check
end

dimension = Integer[]



Nq = No - (2*Ne-2) 
Lzmax = Nq * Ne ÷ 2
for Lz in 0:(Lzmax+1)
    basis = fullhilbertspace(Ne,No, Lz)
    admissibleroots = basis[map(x->isadmissible(x,2,4), basis)]
    push!(dimension, length(admissibleroots))
end
println(dimension)
println("___________________________")
println("L_z sectors:")
println(collect(Lzmax:-1:0))
println("Highest-weight Countings:")
highestweight = dimension[1:end-1] - dimension[2:end]
println(reverse(highestweight))

println("___________________________")
end
