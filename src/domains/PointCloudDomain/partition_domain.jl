# Ported from RadialBasisFiniteDifferences.jl
"""
    partition_domain(casename::String, rbfdeg::Int, polydeg::Int, k::Int, n::Int, num_procs::Int)

Partition the domain for parallel computing.

# Arguments
- `casename::String`: file path to the casename that includes the positions, boundary indices, normals etc.
- `rbfdeg::Int`: degree of the Radial basis function (RBF) for interpolation.
- `polydeg::Int`: degree of the polynomial for approximation.
- `k::Int`: degree of the hyperviscous operator.
- `n::Int`: Stencil size. It is twice the binomial coefficient of `polydeg + 2` choose `2`.
- `num_procs::Int`: number of processors for domain partitioning.

# Returns
This function is complex and involves various steps, including importing data, K-nearest neighbors calculations, graph partitioning, determination of local and halo points, and preparation of halo communication information. 

This function modifies and returns many variables, including but not limited to local and global indices, KNN indices, partition sizes, halo points, and communication indices and lengths. Please refer to the function implementation for detailed outputs.

# Examples
```julia
partition_domain("./medusa/m_points/cyl", 3, 3, 2, 15, 8)
"""
function partition_domain(casename, n, num_procs)
    # positions, interior_idx, boundary_idxs, boundary_normals, normals, boundary_ranges, types = import_medusa(casename)
    positions, interior_idx, boundary_idxs, boundary_normals = read_medusa_file(casename)

    # normals_cyl = normals[boundary_ranges[5]]

    X = X_jl = positions

    # # RBF Parameters
    # k = 2
    # rbfdeg = 3
    # # polydeg = (k * 2) + 1
    # polydeg = 3
    # n = 2 * binomial(polydeg + 2, 2) # Stencil size.
    # # n = 15

    interior = positions[interior_idx]
    Y = X

    ### Overwrite Y closest to X to X value
    # Generate KNN Tree Using HNSW 
    hnsw_y = KDTree(Y)
    idxs_y, dists_y = knn(hnsw_y, X, 1, true)
    idxs_y = [convert.(Int, idxs_y[x]) for x in eachindex(idxs_y)]
    # Proceed with overwrite
    for i in eachindex(X)
        Y[idxs_y[i][1]] = X[i]
    end

    # Generate Knn Tree using NearestNeighbors
    # Addressing a bug where HNSW fails to return point itself 
    # when calculating neighbors.
    hnsw_x = KDTree(X)
    # Find k (approximate) nearest neighbors for each of the queries
    idxs_x, dists_x = knn(hnsw_x, X, n, true)
    # Find single nearest neighbor for each Y point
    idxs_y_x, dists_y_x = knn(hnsw_x, Y, 1)
    # Calculate avg and min distances between points
    _, dists_g = knn(kdtree, medusa_data, 2, true)
    dx_avg = mean(dists_g)[2]
    dx_min = minimum(dists_g)[2]

    ### Partition Graph with Metis
    # num_procs = 8
    # I = deepcopy(idxs_x)
    # V = deepcopy(idxs_x)
    # for i in eachindex(idxs_x)
    #     I[i] .= i
    #     V[i] .= 1
    # end
    # J = deepcopy(idxs_x)
    # C = sparse(vcat(I...), vcat(J...), vcat(V...))
    # G = Metis.graph(C; check_hermitian=false)
    # P = Metis.partition(G, num_procs, alg=:KWAY)
    # scatter(Tuple.(positions), color=P, markersize=10.0, marker=:circle, axis=(aspect=DataAspect(),))

    # Partition Geometrically
    x_extents = maximum([i[1] for i in X]) - minimum([i[1] for i in X])
    y_extents = maximum([i[2] for i in X]) - minimum([i[2] for i in X])
    if x_extents > y_extents
        slab_index = 1
        slab = x_extents / (num_procs)
    else
        slab_index = 2
        slab = y_extents / (num_procs)
    end
    P = Vector{Int64}(undef, length(X))
    extents_min = minimum([i[slab_index] for i in X])
    extents_max = maximum([i[slab_index] for i in X])
    for i in eachindex(P)
        P[i] = floor((X[i][slab_index] - extents_min) / slab) + 1
        if P[i] > num_procs
            P[i] = num_procs
        end
    end

    ### Create Partitions
    partition_size = zeros(Int64, num_procs)
    for i in eachindex(P)
        partition_size[P[i]] += 1
    end
    X_local = Vector{Vector{SVector{2, Float64}}}(undef, num_procs)
    Y_local = Vector{Vector{SVector{2, Float64}}}(undef, num_procs)
    local_to_global_idx = Vector{Vector{Int64}}(undef, num_procs)
    idxs_x_global = Vector{Vector{Vector{Int64}}}(undef, num_procs)
    idxs_y_x_global = Vector{Vector{Vector{Int64}}}(undef, num_procs)
    # halo_local = Vector{Vector{Int64}}(undef, num_procs)
    halo_local = [Int64[] for i in eachindex(partition_size)]
    halo_proc = [Int64[] for i in eachindex(partition_size)]
    halo_global_to_local_idx = Vector{Vector{SVector{2, Int64}}}(undef, num_procs)
    # Initialize the inverse mapping
    global_to_local_idx = Vector{SVector{2, Int64}}(undef, length(P))
    # global_to_local_idx_local = Vector{Vector{Tuple{Int64,Int64}}}(undef, num_procs) # Trying to store global to local but only for points in halo
    for j in eachindex(partition_size)
        i_local = 1
        X_local[j] = Vector{SVector{2, Float64}}(undef, partition_size[j])
        Y_local[j] = Vector{SVector{2, Float64}}(undef, partition_size[j])
        local_to_global_idx[j] = Vector{Int64}(undef, partition_size[j])
        idxs_x_global[j] = Vector{Vector{Int64}}(undef, partition_size[j])
        idxs_y_x_global[j] = Vector{Vector{Int64}}(undef, partition_size[j])
        for i in eachindex(P)
            if P[i] == j
                X_local[j][i_local] = X[i]
                Y_local[j][i_local] = Y[i]
                local_to_global_idx[j][i_local] = i
                idxs_x_global[j][i_local] = idxs_x[i]
                idxs_y_x_global[j][i_local] = idxs_y_x[i]
                i_local += 1
            end
        end
        # Initialize the inverse mapping
        for i in eachindex(local_to_global_idx[j])
            global_idx = local_to_global_idx[j][i]
            local_idx = i
            global_to_local_idx[global_idx] = SVector(local_idx, j)
        end
        # Determine which points are in the halo region
        # halo_local = Int64[]
        for i in eachindex(X_local[j])
            if any(P[idxs_x_global[j][i]] .!= j)
                append!(halo_local[j], idxs_x_global[j][i][P[idxs_x_global[j][i]] .!= j])
                # append!(halo_proc[j], P[idxs_x_global[j][i][P[idxs_x_global[j][i]].!=j]])
            end
        end
        halo_local[j] = unique(halo_local[j]) # Naive implementation of halo region ### Is referencing global X, not local X
        # halo_global_to_local_idx[j] = Vector{Tuple{Int64,Int64}}(undef, length(halo_local[j]))
        # for i in eachindex(halo_local[j])
        #     local_idx_p = global_to_local_idx[halo_local[j][i]]
        #     halo_global_to_local_idx[j][i] = local_idx_p
        # end
    end
    # global_to_local_idx needs to be fully initialized before halo_g2l can be initialized
    for j in eachindex(partition_size)
        halo_global_to_local_idx[j] = Vector{SVector{2, Int64}}(undef,
                                                                length(halo_local[j]))
        for i in eachindex(halo_local[j])
            local_idx_p = global_to_local_idx[halo_local[j][i]]
            halo_global_to_local_idx[j][i] = SVector(local_idx_p...)
            append!(halo_proc[j], local_idx_p[2])
        end
    end
    # sort so that recv is contiguous
    # for j in eachindex(partition_size)
    #     sort!(halo_global_to_local_idx[j], by=x -> x[2])
    # end
    # Issue, may need to also sort halo_local and halo_proc 
    # Need to ensure everything is sorted properly
    p = Vector{Vector{Int64}}(undef, num_procs)
    for j in eachindex(partition_size)
        p[j] = sortperm(halo_global_to_local_idx[j], by = x -> x[2])
        halo_global_to_local_idx[j] = halo_global_to_local_idx[j][p[j]]
        halo_local[j] = halo_local[j][p[j]]
        halo_proc[j] = halo_proc[j][p[j]]
    end

    ### Get Halo Points
    halo_points = Vector{Vector{SVector{2, Float64}}}(undef, num_procs)
    for i in eachindex(halo_global_to_local_idx)
        halo_points[i] = Vector{SVector{2, Float64}}(undef,
                                                     length(halo_global_to_local_idx[i]))
        for j in eachindex(halo_global_to_local_idx[i])
            idx = halo_global_to_local_idx[i][j][1]
            proc = halo_global_to_local_idx[i][j][2]
            halo_points[i][j] = SVector(X_local[proc][idx]...)
        end
    end

    ### Determine which boundary points are in the local X partition
    boundary_local = [[Int64[] for i in eachindex(boundary_idxs)]
                      for j in eachindex(partition_size)]
    for j in eachindex(partition_size)
        for i in eachindex(boundary_idxs)
            for k in eachindex(boundary_idxs[i])
                if j == P[boundary_idxs[i][k]]
                    append!(boundary_local[j][i], boundary_idxs[i][k])
                end
            end
        end
    end
    ### Determine boundary halo interior_idx
    boundary_local_idx = deepcopy(boundary_local)
    # for i in eachindex(local_to_global_idx)
    #     for j in eachindex(local_to_global_idx[i])
    #         for k in eachindex(boundary_local[i])
    #             for l in eachindex(boundary_local[i][k])
    #                 if boundary_local[i][k][l] == local_to_global_idx[i][j]
    #                     boundary_local_idx[i][k][l] = j
    #                 end
    #             end
    #         end
    #     end
    # end
    for i in eachindex(boundary_local_idx)
        for j in eachindex(boundary_local_idx[i])
            for k in eachindex(boundary_local_idx[i][j])
                boundary_local_idx[i][j][k] = global_to_local_idx[boundary_local[i][j][k]][1]
            end
        end
    end
    ### Determine which boundary points are in halo
    boundary_halo = [[Int64[] for i in eachindex(boundary_idxs)]
                     for j in eachindex(partition_size)]
    for j in eachindex(partition_size)
        for i in eachindex(boundary_idxs)
            for k in eachindex(boundary_idxs[i])
                if any(boundary_idxs[i][k] .== halo_local[j])
                    append!(boundary_halo[j][i], boundary_idxs[i][k])
                end
            end
        end
    end
    # boundary_halo_global_to_local = [[[global_to_local_idx[boundary_halo[x][y][z]] for z in eachindex(boundary_halo[x][y])] for y in eachindex(boundary_halo[x])] for x in eachindex(boundary_halo)]
    ### Determine boundary halo interior_idx
    boundary_halo_idx = deepcopy(boundary_halo)
    for i in eachindex(halo_local)
        for j in eachindex(halo_local[i])
            for k in eachindex(boundary_halo[i])
                for l in eachindex(boundary_halo[i][k])
                    if boundary_halo[i][k][l] == halo_local[i][j]
                        boundary_halo_idx[i][k][l] = j
                    end
                end
            end
        end
    end

    ### Partition local normals 
    # boundary_normals = [normals[boundary_ranges[i]] for i in eachindex(boundary_ranges)]
    boundary_normals_local = [[SVector{2, Float64}[] for i in eachindex(boundary_normals)]
                              for j in eachindex(partition_size)]
    for j in eachindex(partition_size)
        for i in eachindex(boundary_idxs)
            for k in eachindex(boundary_idxs[i])
                if j == P[boundary_idxs[i][k]]
                    push!(boundary_normals_local[j][i], boundary_normals[i][k])
                end
            end
        end
    end
    ### Partition halo normals 
    boundary_normals_halo = [[SVector{2, Float64}[] for i in eachindex(boundary_normals)]
                             for j in eachindex(partition_size)]
    for j in eachindex(partition_size)
        for i in eachindex(boundary_idxs)
            for k in eachindex(boundary_idxs[i])
                if any(boundary_idxs[i][k] .== halo_local[j])
                    push!(boundary_normals_halo[j][i], boundary_normals[i][k])
                end
            end
        end
    end

    # Prepare halo comm information
    # Each node needs to know who is sending and who is receiving and how many points
    # Receiving from 
    recv_id = Vector{Vector{Int64}}(undef, num_procs) # May want to use matrix here
    for i in eachindex(recv_id)
        recv_id[i] = unique([halo_global_to_local_idx[i][x][2]
                             for x in eachindex(halo_global_to_local_idx[i])])
    end
    # Sending to
    # send_id = Vector{Vector{Int64}}(undef, num_procs)
    send_id = [Int64[] for i in eachindex(recv_id)]
    for i in eachindex(recv_id)
        for j in eachindex(recv_id[i])
            push!(send_id[recv_id[i][j]], i)
        end
    end
    # Indices received from each proc
    # recv_idx = Vector{Vector{Int64}}(undef, num_procs)
    # May need extra loop for subindex
    recv_idx = [[Int64[] for j in eachindex(recv_id[i])] for i in eachindex(recv_id)]
    for i in eachindex(halo_global_to_local_idx)
        for j in eachindex(halo_global_to_local_idx[i])
            for k in eachindex(recv_id[i])
                if halo_global_to_local_idx[i][j][2] == recv_id[i][k]
                    push!(recv_idx[i][k], halo_global_to_local_idx[i][j][1])
                end
            end
        end
    end
    recv_length = [[length(recv_idx[i][j]) for j in eachindex(recv_id[i])]
                   for i in eachindex(recv_id)]
    # Indices sent to each proc
    send_idx = deepcopy(recv_idx)
    count = ones(Int64, num_procs)
    for i in eachindex(recv_id)
        for j in eachindex(recv_id[i])
            # println("i == recv_id[i][j]: ", i == recv_id[i][j], ", i = $i,", " recv_id = ", recv_id[i][j])
            sending_id = recv_id[i][j]
            send_idx[sending_id][count[sending_id]] = recv_idx[i][j]
            count[sending_id] += 1
        end
    end

    return X_local, Y_local, idxs_x_global, idxs_y_x_global,
           local_to_global_idx, global_to_local_idx,
           halo_points, halo_local, halo_proc, halo_global_to_local_idx,
           send_id, recv_id, send_idx, recv_length,
           boundary_local, boundary_normals_local, boundary_local_idx,
           boundary_halo, boundary_normals_halo, boundary_halo_idx,
           dx_min, dx_avg
end

"""
    scatter_domain(X_partition, local_to_global_idx_partition, 
        halo_points_partition, halo_partition, halo_proc_partition, 
        halo_global_to_local_idx_partition, send_id_p, recv_id_p, 
        send_idx_p, recv_length_p, boundary_partition, boundary_normals_partition, 
        boundary_idx_partition, boundary_halo_partition, boundary_normals_halo_partition, 
        boundary_halo_idx_partition)

Scatter domain data across multiple processors using MPI.jl.

# Arguments
- `X_partition`: A partition of data 'X' to be scattered to each processor.
- `local_to_global_idx_partition`: A mapping from local indices (on each processor) to global indices (across all processors).
- `halo_points_partition`: The partition of 'halo points'. 'Halo' refers to data from neighboring processors required by a processor to perform computations.
- `halo_partition`: A partition of 'halo' data.
- `halo_proc_partition`: A partition of 'halo' data across processors.
- `halo_global_to_local_idx_partition`: A mapping from global indices to local indices for 'halo' data.
- `send_id_p`, `recv_id_p`, `send_idx_p`, `recv_length_p`: Parameters related to sending and receiving messages or data between processors in the parallel computation. 'id' likely refers to processor identifiers, and 'idx' to indices of data to be sent or received.
- Several parameters related to boundaries, which could be the edges of the problem domain in a distributed setting (`boundary_partition`, `boundary_normals_partition`, `boundary_idx_partition`, `boundary_halo_partition`, `boundary_normals_halo_partition`, `boundary_halo_idx_partition`). These might contain information about boundary conditions or boundary cells in a grid computation, for example.

# Returns
The function returns the scattered data after performing the MPI scatter operation. 

# Examples
```julia
# to be filled in with relevant examples
"""
function scatter_domain(X_partition,
                        local_to_global_idx_partition,
                        halo_points_partition, halo_partition, halo_proc_partition,
                        halo_global_to_local_idx_partition,
                        send_id_p, recv_id_p, send_idx_p, recv_length_p,
                        boundary_partition, boundary_normals_partition,
                        boundary_idx_partition,
                        boundary_halo_partition, boundary_normals_halo_partition,
                        boundary_halo_idx_partition)

    ### Scatter data to processors with MPI.jl 
    # X_partition, Y_partition, idxs_x_global_partition, idxs_y_x_global_partition,
    #     local_to_global_idx_partition, global_to_local_idx,
    #     halo_points_partition, halo_partition, halo_proc_partition, halo_global_to_local_idx_partition,
    #     send_id_p, recv_id_p, send_idx_p, recv_length_p,
    #     boundary_partition, boundary_normals_partition, boundary_idx_partition,
    #     boundary_halo_partition, boundary_normals_halo_partition, boundary_halo_idx_partition
    # X

    X = scatter_data(X_partition)
    local_to_global_idx = scatter_data(local_to_global_idx_partition)
    halo_points = scatter_data(halo_points_partition)
    halo_global = scatter_data(halo_partition)
    halo_proc = scatter_data(halo_proc_partition)
    halo_global_to_local_idx = scatter_data(halo_global_to_local_idx_partition)
    boundary_global = scatter_data(boundary_partition)
    boundary_normals = scatter_data(boundary_normals_partition)
    boundary_idx = scatter_data(boundary_idx_partition)
    boundary_halo_global = scatter_data(boundary_halo_partition)
    boundary_normals_halo = scatter_data(boundary_normals_halo_partition)
    boundary_halo_idx = scatter_data(boundary_halo_idx_partition)
    send_id = scatter_data(send_id_p)
    recv_id = scatter_data(recv_id_p)
    send_idx = scatter_data(send_idx_p)
    recv_length = scatter_data(recv_length_p)

    return X, local_to_global_idx,
           halo_points, halo_global, halo_proc, halo_global_to_local_idx,
           boundary_global, boundary_normals, boundary_idx,
           boundary_halo_global, boundary_normals_halo, boundary_halo_idx,
           send_id, recv_id, send_idx, recv_length
end

# Preprocessor
"""
    preprocess(casename, rbfdeg, polydeg, k, n, num_procs)

Preprocesses the data for a specific case, involving partitioning the domain, and then scattering 
the data to the processors.

# Arguments
- `casename::String`: file path to the casename that includes the positions, boundary indices, normals etc.
- `rbfdeg::Int`: degree of the Radial basis function (RBF) for interpolation.
- `polydeg::Int`: degree of the polynomial for approximation.
- `k::Int`: degree of the hyperviscous operator.
- `n::Int`: Stencil size. It is twice the binomial coefficient of `polydeg + 2` choose `2`.
- `num_procs::Int`: number of processors for domain partitioning.

# Process
- Sets up the MPI environment.
- If the processor rank is 0, partitions the domain using the `partition_domain` function.
- For other processors, initializes relevant partitions as undefined vectors.
- Scatters data to processors using the `scatter_domain` function.

# Returns
The scattered data after performing the MPI scatter operation.

# Examples
```julia
# to be filled in with relevant examples
"""
function preprocess(casename, n, num_procs)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    ### partition_domain with function
    if rank == 0
        X_partition, Y_partition, idxs_x_global_partition, idxs_y_x_global_partition,
        local_to_global_idx_partition, global_to_local_idx,
        halo_points_partition, halo_partition, halo_proc_partition, halo_global_to_local_idx_partition,
        send_id_p, recv_id_p, send_idx_p, recv_length_p,
        boundary_partition, boundary_normals_partition, boundary_idx_partition,
        boundary_halo_partition, boundary_normals_halo_partition, boundary_halo_idx_partition, dx_min, dx_avg = partition_domain(casename,
                                                                                                                                 n,
                                                                                                                                 num_procs)
    else
        X_partition = Vector{Vector{SVector{2, Float64}}}(undef, 1)
        local_to_global_idx_partition = Vector{Vector{Int64}}(undef, 1)
        halo_points_partition = Vector{Vector{SVector{2, Float64}}}(undef, 1)
        halo_partition = Vector{Vector{Int64}}(undef, 1)
        halo_proc_partition = Vector{Vector{Int64}}(undef, 1)
        halo_global_to_local_idx_partition = Vector{Vector{SVector{2, Int64}}}(undef, 1)
        boundary_partition = Vector{Vector{Vector{Int64}}}(undef, 1)
        boundary_normals_partition = Vector{Vector{Vector{SVector{2, Float64}}}}(undef, 1)
        boundary_idx_partition = Vector{Vector{Vector{Int64}}}(undef, 1)
        boundary_halo_partition = Vector{Vector{Vector{Int64}}}(undef, 1)
        boundary_normals_halo_partition = Vector{Vector{Vector{SVector{2, Float64}}}}(undef,
                                                                                      1)
        boundary_halo_idx_partition = Vector{Vector{Vector{Int64}}}(undef, 1)
        send_id_p = Vector{Vector{Int64}}(undef, 1)
        recv_id_p = Vector{Vector{Int64}}(undef, 1)
        send_idx_p = Vector{Vector{Vector{Int64}}}(undef, 1)
        recv_length_p = Vector{Vector{Int64}}(undef, 1)
    end

    ### Scatter data to processors with MPI.jl 
    X, local_to_global_idx,
    halo_points, halo_global, halo_proc, halo_global_to_local_idx,
    boundary_global, boundary_normals, boundary_idx,
    boundary_halo_global, boundary_normals_halo, boundary_halo_idx,
    send_id, recv_id, send_idx, recv_length = scatter_domain(X_partition,
                                                             local_to_global_idx_partition,
                                                             halo_points_partition,
                                                             halo_partition,
                                                             halo_proc_partition,
                                                             halo_global_to_local_idx_partition,
                                                             send_id_p,
                                                             recv_id_p,
                                                             send_idx_p,
                                                             recv_length_p,
                                                             boundary_partition,
                                                             boundary_normals_partition,
                                                             boundary_idx_partition,
                                                             boundary_halo_partition,
                                                             boundary_normals_halo_partition,
                                                             boundary_halo_idx_partition)

    return X, local_to_global_idx,
           halo_points, halo_global, halo_proc, halo_global_to_local_idx,
           boundary_global, boundary_normals, boundary_idx,
           boundary_halo_global, boundary_normals_halo, boundary_halo_idx,
           send_id, recv_id, send_idx, recv_length, dx_min, dx_avg
end