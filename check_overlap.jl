include("/home/trung/_qhe-julia/FQH_state_v2.jl")
using .FQH_states

using ArgMacros
using LinearAlgebra
using BenchmarkTools

function main()

@inlinearguments begin
	@argumentoptional String basis1 "--basis1"
	@argumentoptional String basis2 "--basis2" 
	@argumentflag decimal1 "--decimal1"
	@argumentflag decimal2 "--decimal2"
	@argumentoptional Int No "--n_orb" "-o"
	@argumentdefault String "none" normalize1 "--normalize1"
	@argumentdefault String "none" normalize2 "--normalize2"
	@positionalrequired String state1
	@positionalrequired String state2
	end

if state1 == nothing || state2 == nothing
	println("Two states must be specified.")
	return
end

if !isfile(state1)
	println("The specified file '$(state1)' doesn't exist.")
	return
end

if !isfile(state2)
	println("The specified file '$(state2)' doesn't exist.")
	return
end

if (basis1 != nothing || state2 != nothing || decimal ) && No == nothing
	println("For a decimal format, the number of orbital must be specified.")
end

if basis1 == nothing
	if decimal1
		wf1 = readwfdec(state1,No)
	else
		wf1 = readwf(state1)
	end
else
	wf1 = readwf(basis1,state1,No)
end

if basis2 == nothing
	if decimal2
		wf2 = readwfdec(state2,No)
	else
		wf2 = readwf(state2)
	end
else
	wf2 = readwf(basis2,state2,No)
end

if lowercase(normalize1)=="sphere"
	println("Normalize wavefunction 1 on the sphere")
	wf1 = sphere_normalize(wf1)
elseif lowercase(normalize1)=="disk"
	println("Normalize wavefunction 1 on the disk")
	wf1 = disk_normalize(wf1)
end

if lowercase(normalize2)=="sphere"
	println("Normalize wavefunction 2 on the sphere")
	wf2 = sphere_normalize(wf2)
elseif lowercase(normalize2)=="disk"
	println("Normalize wavefunction 2 on the disk")
	wf2 = disk_normalize(wf2)
end

ov = wf1 ⋅ wf2
println("Overlap = $(ov)")
println("|Overlap| = $(abs(ov))")
return

end

@time main()