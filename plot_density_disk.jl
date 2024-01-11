include("FQH_state_v2.jl")
include("Density.jl")
using .FQH_states
using .ParticleDensity
using BenchmarkTools
using ArgMacros
using Plots

ENV["GKSwstype"] = "100" # Disconnect from Display

function main()
@inlinearguments begin
    @argumentrequired String fname "-f" "--filename"
end

state = readwf(fname)
No = length(state.basis[1])
R_max = sqrt(2*No)+0.2

println("Check norm = $(wfnorm(state))")

x = -R_max:0.05:R_max
y = -R_max:0.05:R_max

N = length(x)

Y = repeat(y, inner=(1,N))
X = collect(transpose(Y))

Z = 0.5(X+Y*im)
single_particle = [single_particle_state_disk.(Z,m) for m in 0:length(state.basis[1])]

D = dim(state)

den = zeros(size(Z))
@time begin
for i in 1:D
    print("\r$i\t")
    for j in i:D
        coef = 2^(i!=j) * conj(state.coef[i]) * state.coef[j]
        density_element_gen!(den, coef, state.basis[i], state.basis[j], single_particle)
    end
end
end

println("-----")
#den = @time get_density_disk(state, X, Y)

open("$(fname)_density.dat", "w") do f
    for i in 1:N, j in 1:N
        write(f, "$(X[i,j])\t$(Y[i,j])\t$(den[i,j])\n")
    end
end


#p = plot(heatmap(z=density, aspect_ratio=:equal)) 

p = heatmap(x, y, den, aspect_ratio=:equal)
savefig(p, "$(fname)_density.svg")
end

main()