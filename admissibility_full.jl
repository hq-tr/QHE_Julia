include("HilbertSpace.jl")
using Main.HilbertSpaceGenerator

include("FQH_state_v2.jl")
using Main.FQH_states

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

println("Input N_orb: ")
No = parse(Int, readline())

println("Input N_el: ")
Ne = parse(Int, readline())

println("(k,r)-admissibility: no more than k electrons within any r consecutive orbitals")
println("Input k:")
k  = parse(Int, readline())
println("Input r:")
r  = parse(Int, readline())

println("Output file name:")
fname = readline()

basis = fullhilbertspace(Ne,No)
admissibleroots = basis[map(x->isadmissible(x,k,r), basis)]
zero_coefs = zeros(length(admissibleroots))

dummy = FQH_state(admissibleroots, zero_coefs)
printwf(dummy;fname=fname)