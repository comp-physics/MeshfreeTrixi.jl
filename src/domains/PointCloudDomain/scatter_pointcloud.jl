# using MPI, StaticArrays

# For scattering points
function scatter_data(data::Vector{Vector{SVector{2,T}}}, root::Int=0) where {T<:Number}
    # MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_processes = MPI.Comm_size(comm)

    # if rank == 0
    #     println("Scattering points")
    # end

    if rank == root
        # Convert each SVector to a Vector and create a matrix from the vectors
        sendbufs = [hcat([convert(Vector, v) for v in inner_vec]...) for inner_vec in data]
        # Concatenate all matrices into one matrix
        sendbuf = hcat(sendbufs...)
        # Counts and displacements for Scatterv
        # sendcounts = length.(data) .* length(data[1][1])
        sendcounts = length.(data) .* 2
        sendcount_buf = MPI.UBuffer(sendcounts, 1)
        displs = vcat(0, cumsum(sendcounts[1:end-1]))
        sendbuf = MPI.VBuffer(sendbuf, sendcounts, displs)
    else
        sendbuf = nothing
        sendcount_buf = MPI.UBuffer(nothing)
    end

    # Scatter the counts and displacements
    recvcount = MPI.Scatter(sendcount_buf, Int64, root, comm)
    # for i = 0:num_processes-1
    #     if rank == i
    #         # @show rank recvdata
    #         println("Process $rank received count ", recvcount)
    #     end
    #     MPI.Barrier(comm)
    # end

    # Each process will receive a variable number of SVectors (as many as its rank)
    # recvcount = rank * 2  # Each SVector{2, Float64} has 2 elements
    recvcount_col = Int(recvcount / 2)
    recvbuf = Array{T}(undef, 2, recvcount_col)
    # for i = 0:num_processes-1
    #     if rank == i
    #         # @show rank recvdata
    #         println("Process $rank recv buffer ", recvbuf)
    #     end
    #     MPI.Barrier(comm)
    # end
    recvbuf = MPI.Buffer(recvbuf)

    # Scatter the data
    MPI.Scatterv!(sendbuf, recvbuf, comm; root=root)

    # Plot the received data
    # for i = 0:num_processes-1
    #     if rank == i
    #         # @show rank recvdata
    #         println("Process $rank received raw ", recvbuf.data)
    #     end
    #     MPI.Barrier(comm)
    # end

    # Convert back to SVectors after receiving
    recvdata = [SVector{2,T}(recvbuf.data[:, i]) for i in 1:recvcount_col]

    # for i = 0:num_processes-1
    #     if rank == i
    #         # @show rank recvdata
    #         println("Process $rank received ", recvdata)
    #     end
    #     MPI.Barrier(comm)
    # end
    # println("Process $rank received ", recvdata)

    return recvdata
end

# For scattering indices
function scatter_data(data::Vector{Vector{T}}, root::Int=0) where {T<:Number}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_processes = MPI.Comm_size(comm)

    if rank == root
        sendbufs = [hcat(inner_vec...) for inner_vec in data]
        sendbuf = hcat(sendbufs...)
        sendcounts = length.(data)
        sendcount_buf = MPI.UBuffer(sendcounts, 1)
        displs = vcat(0, cumsum(sendcounts[1:end-1]))
        sendbuf = MPI.VBuffer(sendbuf, sendcounts, displs)
    else
        sendbuf = nothing
        sendcount_buf = MPI.UBuffer(nothing)
    end

    recvcount = MPI.Scatter(sendcount_buf, Int64, root, comm)
    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received count ", recvcount)
    #     end
    #     MPI.Barrier(comm)
    # end

    recvbuf = Array{T}(undef, recvcount)
    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank recv buffer ", recvbuf)
    #     end
    #     MPI.Barrier(comm)
    # end
    recvbuf = MPI.Buffer(recvbuf)

    MPI.Scatterv!(sendbuf, recvbuf, comm; root=root)

    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received raw ", recvbuf.data)
    #     end
    #     MPI.Barrier(comm)
    # end

    recvdata = recvbuf.data

    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received ", recvdata)
    #     end
    #     MPI.Barrier(comm)
    # end

    return recvdata
end

# For scattering neighbor indices
function scatter_data(data::Vector{Vector{Vector{T}}}, root::Int=0) where {T<:Number}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_processes = MPI.Comm_size(comm)

    if rank == root
        # Flatten each inner vector into a single vector and calculate the displacements
        sendbufs = vcat([vcat(inner_vec...) for inner_vec in data]...)
        sendcounts = [length(vcat(inner_vec...)) for inner_vec in data]
        sendcount_buf = MPI.UBuffer(sendcounts, 1)
        displs = vcat(0, cumsum(sendcounts[1:end-1]))
        sendbuf = MPI.VBuffer(sendbufs, sendcounts, displs)
        # Scatter for reconstructing inner vectors
        pointcounts = length.(data)
        pointcounts_buf = MPI.UBuffer(pointcounts, 1)
        innercounts = [length.(data[i]) for i in eachindex(data)]
        innercounts_flat = vcat(innercounts...)
        innercounts_counts = length.(innercounts)
        innercounts_displs = vcat(0, cumsum(innercounts_counts[1:end-1]))
        innercounts_buf = MPI.VBuffer(innercounts_flat, innercounts_counts, innercounts_displs)
        # innercounts_buf = MPI.UBuffer(vcat(innercounts...), 1)
    else
        sendbuf = nothing
        sendcount_buf = MPI.UBuffer(nothing)
        pointcounts_buf = MPI.UBuffer(nothing)
        innercounts_buf = nothing
    end

    recvcount = MPI.Scatter(sendcount_buf, Int64, root, comm)
    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received count ", recvcount)
    #     end
    #     MPI.Barrier(comm)
    # end

    recvbuf = Array{T}(undef, recvcount)
    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank recv buffer ", recvbuf)
    #     end
    #     MPI.Barrier(comm)
    # end
    recvbuf = MPI.Buffer(recvbuf)

    MPI.Scatterv!(sendbuf, recvbuf, comm; root=root)

    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received raw ", recvbuf.data)
    #     end
    #     MPI.Barrier(comm)
    # end

    # Reconstruct the inner vectors
    pointcounts_recv = MPI.Scatter(pointcounts_buf, Int64, root, comm)
    innercounts_recv = Array{Int64}(undef, pointcounts_recv)
    innercounts_recv = MPI.Buffer(innercounts_recv)
    MPI.Scatterv!(innercounts_buf, innercounts_recv, comm; root=root)
    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received ", innercounts_recv.data)
    #     end
    #     MPI.Barrier(comm)
    # end

    # Reconstruct vector from flattened vector
    # recvdata = [recvbuf.data[i] for i in 1:recvcount]
    # recvdata = [recvbuf.data[innercounts_recv.data[i]] for i in 1:pointcounts_recv]
    idx = 1
    recvdata = Vector{Vector{T}}(undef, length(innercounts_recv.data))
    for i in 1:length(innercounts_recv.data)
        recvdata[i] = recvbuf.data[idx:(idx+innercounts_recv.data[i]-1)]
        idx += innercounts_recv.data[i]
    end

    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received ", recvdata)
    #     end
    #     MPI.Barrier(comm)
    # end

    return recvdata
end

# For scattering boundary normals, SVector{2, Float64}
function scatter_data(data::Vector{Vector{Vector{SVector{2,T}}}}, root::Int=0) where {T<:Number}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_processes = MPI.Comm_size(comm)

    if rank == root
        # Convert each SVector to a Vector and create a matrix from the vectors
        sendbufs = [hcat([convert(Vector, v) for v in inner_inner_vec]...) for inner_vec in data for inner_inner_vec in inner_vec]
        # Concatenate all matrices into one matrix
        filtered_sendbufs = filter(x -> !isempty(x), sendbufs)
        sendbuf = hcat(filtered_sendbufs...)
        # sendcounts = [length(vcat(inner_vec...)) for inner_vec in data] .* length(data[1][1][1])
        sendcounts = [length(vcat(inner_vec...)) for inner_vec in data] .* 2
        sendcount_buf = MPI.UBuffer(sendcounts, 1)
        displs = vcat(0, cumsum(sendcounts[1:end-1]))
        sendbuf = MPI.VBuffer(sendbuf, sendcounts, displs)
        # Scatter for reconstructing inner vectors
        pointcounts = length.(data)
        pointcounts_buf = MPI.UBuffer(pointcounts, 1)
        innercounts = [length.(data[i]) for i in eachindex(data)]
        innercounts_flat = vcat(innercounts...)
        innercounts_counts = length.(innercounts)
        innercounts_displs = vcat(0, cumsum(innercounts_counts[1:end-1]))
        innercounts_buf = MPI.VBuffer(innercounts_flat, innercounts_counts, innercounts_displs)
        # innercounts_buf = MPI.UBuffer(vcat(innercounts...), 1)
    else
        sendbuf = nothing
        sendcount_buf = MPI.UBuffer(nothing)
        pointcounts_buf = MPI.UBuffer(nothing)
        innercounts_buf = nothing
    end

    recvcount = MPI.Scatter(sendcount_buf, Int64, root, comm)
    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received count ", recvcount)
    #     end
    #     MPI.Barrier(comm)
    # end

    # Each process will receive a variable number of SVectors (as many as its rank)
    # recvcount = rank * 2  # Each SVector{2, Float64} has 2 elements
    recvcount_col = Int(recvcount / 2)
    recvbuf = Array{T}(undef, 2, recvcount_col)
    # recvbuf = Array{Int64}(undef, recvcount)
    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank recv buffer ", recvbuf)
    #     end
    #     MPI.Barrier(comm)
    # end
    recvbuf = MPI.Buffer(recvbuf)

    MPI.Scatterv!(sendbuf, recvbuf, comm; root=root)

    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received raw ", recvbuf.data)
    #     end
    #     MPI.Barrier(comm)
    # end

    # Reconstruct the inner vectors
    pointcounts_recv = MPI.Scatter(pointcounts_buf, Int64, root, comm)
    innercounts_recv = Array{Int64}(undef, pointcounts_recv)
    innercounts_recv = MPI.Buffer(innercounts_recv)
    MPI.Scatterv!(innercounts_buf, innercounts_recv, comm; root=root)
    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received ", innercounts_recv.data)
    #     end
    #     MPI.Barrier(comm)
    # end

    # Reconstruct vector from flattened vector
    # idx = 1
    # recvdata = Vector{Vector{Int64}}(undef, length(innercounts_recv.data))
    # for i in 1:length(innercounts_recv.data)
    #     recvdata[i] = recvbuf.data[idx:(idx+innercounts_recv.data[i]-1)]
    #     idx += innercounts_recv.data[i]
    # end
    idx = 1
    recvdata = Vector{Vector{SVector{2,T}}}(undef, length(innercounts_recv.data))
    for i in 1:length(innercounts_recv.data)
        recvdata[i] = [SVector{2,T}(recvbuf.data[1, idx+step], recvbuf.data[2, idx+step]) for step in 0:(innercounts_recv.data[i]-1)]
        idx += innercounts_recv.data[i]
    end

    # for i = 0:num_processes-1
    #     if rank == i
    #         println("Process $rank received ", recvdata)
    #     end
    #     MPI.Barrier(comm)
    # end

    return recvdata
end

# Scatter tests
# MPI.Init()
# data = [[SVector{2,Float64}(i, i) for _ in 1:i] for i in 1:MPI.Comm_size(MPI.COMM_WORLD)]
# scatter_data(data)

# data = [[i for _ in 1:i] for i in 1:MPI.Comm_size(MPI.COMM_WORLD)]
# scatter_data(data)

# data = [[[j + 5 * (i - 1) for j in 1:i] for i in 1:p] for p in 1:MPI.Comm_size(MPI.COMM_WORLD)]
# scatter_data(data)

# if MPI.Comm_rank(MPI.COMM_WORLD) == 0
#     println("\nNew test\n")
# end
# data = [[[SVector{2,Float64}(j, j) for _ in 1:j] for j in 1:i] for i in 1:MPI.Comm_size(MPI.COMM_WORLD)]
# scatter_data(data)

# MPI.Finalize()