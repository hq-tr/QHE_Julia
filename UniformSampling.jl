module UniformSampling

function circle_sampling(center_x::Float64,center_y::Float64, radius::Float64, num::Int)
	r = radius * sqrt.(rand(num))
	θ = 2π * rand(num)
	x = center_x + r*cos.(θ)
	y = center_y + r*sin.(θ)
	return x,y
end

function ellipse_sampling(center_x::Float04, center_y::Float64, major::Float64, minor::Float64, rot_angle::Float64, num::Int)
	r = sqrt.(rand(num))
	θ = 2π * rand(num)
	x1 = major * r * cos.(θ)
	y1 = minor * r * sin.(θ)
	x = center_x + x1 * cos(rot_angle) - y1 * sin(rot_angle)
	y = center_y + x1 * sin(rot_angle) + y1 * cos(rot_angle)
	return x, y
end

end
