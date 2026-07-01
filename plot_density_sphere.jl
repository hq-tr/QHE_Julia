include("/home/trung/_qhe-julia/FQH_state_v2.jl")
include("/home/trung/_qhe-julia/Density.jl")
using .FQH_states
using .ParticleDensity
using BenchmarkTools
using ArgMacros
using Plots; plotlyjs()

ENV["GKSwstype"] = "100" # Disconnect from Display

function main()
@inlinearguments begin
    @argumentrequired String fname "-f" "--filename"
    @argumentflag dec_format "--decimal"
    @argumentoptional Int No "--n_orb" "-o"
    @argumentoptional String basisname "-b" "--basis"
    @argumentflag lineplot "--line"
    @argumentoptional String outputname "-O" "--output"
    @argumentflag normalize "-N" "--normalize"
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

R_max = sqrt(2*No)+0.2

println("Check norm = $(wfnorm(state))")

if normalize
    println("--------------")
    println("Wavefunction will be normalized on the sphere.")
    state = sphere_normalize(state)
    println("Check norm = $(wfnorm(state))")
    println("--------------")
end

if lineplot
    # Create the plot range
    θ = LinRange(0,π,200)
    if !check_Lz_eigenstate(state)
        # If the state is eigenstate of Lz, only calculate density along ϕ=0
        println("NOTE: Line plot was selected, but the state is NOT an Lz eigenstate.")
        println("Line plot will be made long the longitude ϕ=0")
    end

    # Calculate density
    @time den_line = get_density_sphere(state, collect(θ), zeros(size(θ)))
    

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
        @time den_line = get_density_sphere(state, collect(θ), zeros(size(θ)))
        den = repeat(den_line,inner=(1,N₂))
    else
        #den = zeros(size(Z))
        @time den = get_density_sphere(state, Θ, Φ)
    end

    println("-----")
    #den = @time get_density_disk(state, X, Y)

    # Save density data
    if outputname == nothing
        outputname = "$(fname)_density_sphere"
    end
    open("$(outputname).dat", "w") do f
        for i in 1:N₁, j in 1:N₂
            write(f, "$(Θ[i,j])\t$(Φ[i,j])\t$(den[i,j])\n")
        end
    end

    #p = plot(heatmap(z=density, aspect_ratio=:equal)) 

    xx = sin.(Θ) .* cos.(Φ)
    yy = sin.(Θ) .* sin.(Φ)
    zz = cos.(Θ)

    p = surface(xx, yy, zz, fill_z = den, size=(600,600))
    savefig(p, "$(outputname).html")
end

end # end function main()

@time main()