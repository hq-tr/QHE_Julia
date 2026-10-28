# Version 2 of plot_density_sphere.jl with the same command-line options.
#
# Instead of summing over all pairs of basis states at every grid point, the one-body density matrix
#     D_ab = <Ψ| c†_a c_b |Ψ>
# is computed once, and the density on the grid is then
#     ρ(θ,ϕ) = Σ_ab D_ab conj(ψ_a(θ,ϕ)) ψ_b(θ,ϕ)
# D is built by acting with c†_a c_b on each basis state and looking up the result in a dictionary,
# which costs O(dim · Nₑ · Nₒ) instead of O(dim²).

include("/home/trung/_qhe-julia/FQH_state_v2.jl")
include("/home/trung/_qhe-julia/Density.jl")
using .FQH_states
using .ParticleDensity
using BenchmarkTools
using ArgMacros
using Plots; plotlyjs()

ENV["GKSwstype"] = "100" # Disconnect from Display

# One-body density matrix D_ab = <Ψ| c†_a c_b |Ψ> (orbital indices a,b start from 1)
# Fermion sign convention follows density_element_gen in Density.jl
function one_body_density_matrix(state::AbstractFQH_state)
    No = length(state.basis[1])
    D = zeros(ComplexF64, No, No)
    index = Dict(b => i for (i,b) in enumerate(state.basis))
    for (Mu, c_Mu) in zip(state.basis, state.coef)
        occupied = findall(Mu)
        empty    = findall(.!Mu)
        for b in occupied
            D[b,b] += abs2(c_Mu)
            sign_b = count(Mu[1:b-1])
            for a in empty
                Lam = copy(Mu)
                Lam[b] = false
                Lam[a] = true
                i = get(index, Lam, 0)
                if i > 0
                    sign_a = count(Lam[1:a-1])
                    D[a,b] += (-1)^(sign_a+sign_b) * conj(state.coef[i]) * c_Mu
                end
            end
        end
    end
    return D
end

# Density on the sphere from the one-body density matrix, on arrays θ, ϕ of the same shape
function get_density_sphere_v2(state::AbstractFQH_state, θ::Array{T} where T<:Number, ϕ::Array{T} where T<:Number)
    No = length(state.basis[1])
    S  = 0.5(No-1)
    D  = one_body_density_matrix(state)
    # Φ[r,m+1] = ψ_m(r) for orbital m (Lz = S-m)
    Φ = hcat([vec(single_particle_state_sphere(θ, ϕ, S, S-m)) for m in 0:(No-1)]...)
    den = real.(vec(sum(conj.(Φ) .* (Φ * transpose(D)), dims=2)))
    return reshape(den, size(θ))
end

function main()
@inlinearguments begin
    @argumentrequired String fname "-f" "--filename"
    @argumentflag dec_format "--decimal"
    @argumentoptional Int No "--n_orb" "-o"
    @argumentoptional String basisname "-b" "--basis"
    @argumentflag lineplot "--line"
    @argumentoptional String outputname "-O" "--output"
    @argumentflag normalize "-N" "--normalize"
    @argumentflag orbital_density "--orbital"
end

if basisname == nothing
    if dec_format
        if No == nothing
            println("Decimal format requires number of orbital (Nₒ) input.")
            print("Input Nₒ: ")
            No = parse(Int,readline())
        end
        state = readwfdec(fname,No)
    else
        state = readwf(fname)
        if No == nothing
            No = length(state.basis[1])
        end
    end
else
    if No == nothing
        println("Decimal format requires number of orbital (Nₒ) input.")
        print("Input Nₒ: ")
        No = parse(Int,readline())
    end
    state = readwf(basisname,fname,No)
end

println("Check norm = $(wfnorm(state))")

if normalize
    println("--------------")
    println("Wavefunction will be normalized on the sphere.")
    state = sphere_normalize(state)
    println("Check norm = $(wfnorm(state))")
    println("--------------")
end

if orbital_density
    density = zeros(No)
    for (b,c) in zip(state.basis,state.coef)
        density .+= b.*abs2.(c)
    end

    # Save density data
    if outputname == nothing
        outputname = "$(fname)_density_sphere_orbital"
    end
    open("$(outputname).dat", "w") do f
        for (datax,datay) in enumerate(density)
            write(f,"$(datax-1)\t$(datay)\n")
        end
    end

    # Plot
    p = plot(density,markershape=:xcross,linestyle=:dash)
    xlabel!("Orbital number")
    ylabel!("Density")
    savefig("$(outputname).svg")
elseif lineplot
    # Create the plot range
    θ = LinRange(0,π,200)
    if !check_Lz_eigenstate(state)
        # If the state is eigenstate of Lz, only calculate density along ϕ=0
        println("NOTE: Line plot was selected, but the state is NOT an Lz eigenstate.")
        println("Line plot will be made long the longitude ϕ=0")
    end

    # Calculate density
    @time den_line = get_density_sphere_v2(state, collect(θ), zeros(size(θ)))


    # Save density data
    if outputname == nothing
        outputname = "$(fname)_density_sphere_line"
    end
    open("$(outputname).dat", "w") do f
        for (datax,datay) in zip(θ,den_line)
            write(f,"$(datax)\t$(datay)\n")
        end
    end

    # Plot
    p = plot(θ,den_line)
    xlabel!("θ")
    ylabel!("ρ(θ,ϕ=0)")
    savefig("$(outputname).svg")
else
    # Create the plot range
    θ = LinRange(0,π,40)
    ϕ = LinRange(0,2π, 80)

    N₁ = length(θ)
    N₂ = length(ϕ)

    # Create the θ-ϕ mesh grid
    Θ = repeat(θ, inner=(1,N₂))
    Φ = repeat(ϕ', inner=(N₁,1))

    # Calculate density
    if check_Lz_eigenstate(state)
        # If the state is eigenstate of Lz, only calculate density along ϕ=0
        println("NOTE: The state is an Lz eigenstate.")
        @time den_line = get_density_sphere_v2(state, collect(θ), zeros(size(θ)))
        den = repeat(den_line,inner=(1,N₂))
    else
        @time den = get_density_sphere_v2(state, Θ, Φ)
    end

    println("-----")

    # Save density data
    if outputname == nothing
        outputname = "$(fname)_density_sphere"
    end
    open("$(outputname).dat", "w") do f
        for i in 1:N₁, j in 1:N₂
            write(f, "$(Θ[i,j])\t$(Φ[i,j])\t$(den[i,j])\n")
        end
    end

    xx = sin.(Θ) .* cos.(Φ)
    yy = sin.(Θ) .* sin.(Φ)
    zz = cos.(Θ)

    p = surface(xx, yy, zz, fill_z = den, size=(600,600))
    savefig(p, "$(outputname).html")
end

end # end function main()

@time main()
