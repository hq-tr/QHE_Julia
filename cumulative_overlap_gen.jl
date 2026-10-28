include("/home/trung/_qhe-julia/FQH_state_v2.jl")
include("/home/trung/_qhe-julia/HilbertSpace.jl")
include("/home/trung/_qhe-julia/Misc.jl")
using .FQH_states
using .HilbertSpaceGenerator
using .MiscRoutine

using ArgMacros

function main()
	@inlinearguments begin
		@argumentrequired String filename "--file" "-f"
		@argumentoptional String basisname "--basis" "-b"
		@argumentoptional String normalize "--normalize"
		@argumentflag decimal "--decimal"
		@argumentoptional Int n_orb "--n_orb" "-o"
		@argumentrequired String dirname "--directory" "-d"
		@argumentoptional String outputfile "--output"
		@argumentflag decimaldirectory "--decimal-directory"
	end

	# check directory of model states
	modelfiles = readdir(dirname)
	if length(modelfiles) == 0
		println("Specified directory is empty.\n\nTerminating.\n")
		return
	else
		println("Directory contains $(length(modelfiles)) file(s).")
	end

	# read trial state
	if isfile(filename)
		if basisname == nothing 
			# No basis supplied. File is a stand-alone wavefunction
			if decimal
				if n_orb != nothing
					state = readwfdec(filename,n_orb)
				else
					println("For a wavefunction in decimal format, the number of orbitals n_orb must be specified.")
					println("\nTerminating.\n")
					return
				end
			else
				state = readwf(filename)
			end
		else
			if n_orb != nothing
				state = readwf(basisname,filename,n_orb)
			else
				println("For a wavefunction with a separate basis file, the number of orbitals n_orb must be specified.")
				println("\nTerminating.\n")
				return
			end
		end
	else
		println("Input file $(filename) not found.\n\nTerminating.\n")
		return
	end

	if normalize != nothing
		if lowercase(normalize) == "sphere"
			state = sphere_normalize(state)
		elseif lowercase(normalize) == "disk"
			state = disk_normalize(state)
		end
	end
	
	# read directory of model states and take overlaps
	overlap_sq = zeros(length(modelfiles)) # A list of squared overlap with each model state

	Threads.@threads for (i,dfile) in collect(enumerate(modelfiles))
		fullpath   = "$(dirname)/$(dfile)"
		print("\rReading $(fullpath) \t File $i out of $(length(modelfiles))\t\t")
		if !decimaldirectory
			modelstate = readwf(fullpath;verbose=false) # Here, assuming model states are in binary format
		else
			if n_orb != nothing
				modelstate = readwfdec(fullpath,n_orb;verbose=false)
			else
				println("For a wavefunction with a separate basis file, the number of orbitals n_orb must be specified.")
				println("\nTerminating.\n")
				return
			end
		end
		ov2 = abs2(overlap(modelstate,state))
		overlap_sq[i] = ov2
	end
	total_overlap_sq = sum(overlap_sq)
	println("Done!")
	println("\n----")
	println("TOTAL OVERLAP = $(sqrt(total_overlap_sq)).")
	println("Maximum overlap with a single state = $(sqrt(maximum(overlap_sq)))")
	println("--------------------\n")

	if outputfile != nothing
		open(outputfile,"w+") do f
			write(f,"total squared overlap\t$(total_overlap_sq)\n")
			write(f,"total overlap \t$(sqrt(total_overlap_sq))\n")
			for (b,o) in zip(modelfiles,overlap_sq)
				write(f,"$b\t$o\n")
			end
		end
		println("Saved to file $(outputfile)")
	end
end

@time main()