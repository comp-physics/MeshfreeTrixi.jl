### Function for importing Medusa point data
function read_medusa_file(casename)
    ## Read in data
    positions = readdlm(casename * "_positions.txt", ',', Float64, '\n')
    types = vec(readdlm(casename * "_types.txt", ' ', Int64, '\n'))
    boundary_idx = vec(readdlm(casename * "_boundary.txt", ' ', Int64, '\n') .+ 1)
    interior_idx = vec(readdlm(casename * "_interior.txt", ' ', Int64, '\n') .+ 1)
    normals = readdlm(casename * "_normals.txt", ',', Float64, '\n')

    ## Create vector of vector of ints for each boundary type
    num_bound = -1 * minimum(types)
    boundary_idxs = Vector{Vector{Int64}}(undef, num_bound)
    boundary_normals = Vector{Vector{SVector{2, Float64}}}(undef, num_bound)
    for i in 1:num_bound
        boundary_idxs[i] = Int64[]
        boundary_normals[i] = SVector{2, Float64}[]
    end
    for j in eachindex(boundary_idx)
        for i in 1:num_bound
            if types[boundary_idx[j]] == -i
                push!(boundary_idxs[i], boundary_idx[j])
                push!(boundary_normals[i], SVector{2, Float64}(normals[j, :]))
            end
        end
    end
    ## Remove empty elements 
    boundary_idxs = boundary_idxs[.!isempty.(boundary_idxs)]

    interior = positions[interior_idx, :]
    # scatter(interior[:, 1], interior[:, 2], color=:black, markersize=10.0, marker=:circle, axis=(aspect=DataAspect(),))

    ## Plot boundaries
    for i in eachindex(boundary_idxs)
        boundary = positions[boundary_idxs[i], :]
        # scatter!(boundary[:, 1], boundary[:, 2], markersize=20.0, marker=:circle)
    end

    ## Boundary Ranges
    boundary_ranges_ = length.(boundary_idxs)
    boundary_ranges = Vector{Vector{Int64}}(undef, length(boundary_ranges_))
    for i in eachindex(boundary_ranges_)
        if i == 1
            boundary_ranges[i] = 1:boundary_ranges_[i]
        else
            boundary_ranges[i] = (boundary_ranges[i - 1][end] + 1):(boundary_ranges[i - 1][end] + boundary_ranges_[i])
        end
    end

    ## Convert positions matrix to vector of vectors
    # positions = Vector{Vector{Float64}}(undef, size(positions, 1))
    # positions = [positions[i, :] for i in 1:size(positions, 1)]
    # position_out = Array{SVector{2},1}(undef, length(positions))
    position_out = copy(vec(reinterpret(SVector{2, Float64}, positions')))

    ## Convert normals matrix to vector of vectors
    normal_out = copy(vec(reinterpret(SVector{2, Float64}, normals')))

    # positions, interior_idx, boundary_idxs, normals, boundary_ranges, types = import_medusa(casename)
    ### Partition local normals 
    # boundary_normals = [normal_out[boundary_ranges[i]] for i in eachindex(boundary_ranges)]

    return position_out, interior_idx, boundary_idxs, boundary_normals, normal_out,
           boundary_ranges, types
end
