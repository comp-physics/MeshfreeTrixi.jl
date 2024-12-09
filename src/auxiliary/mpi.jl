"""
    ode_norm(u, t)

Implementation of the weighted L2 norm of Hairer and Wanner used for error-based
step size control in OrdinaryDiffEq.jl. This function is aware of MPI and uses
global MPI communication when running in parallel.

You must pass this function as a keyword argument
`internalnorm=ode_norm`
to OrdinaryDiffEq.jl's `solve` when using error-based step size control with MPI
parallel execution of Trixi.jl.

See the "Advanced Adaptive Stepsize Control" section of the [documentation](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).
"""
function Trixi.ode_norm(u::SVector, t)
    local_sumabs2 = recursive_sum_abs2(u) # sum(abs2, u)
    local_length = recursive_length(u)    # length(u)
    return sqrt(local_sumabs2 / local_length)
end

"""
    perform_halo_update!(u::Union{Array{Float64,1},SubArray{Float64,1}}, halo::Union{Array{Float64,1},SubArray{Float64,1}}, send_id, recv_id, send_idx, recv_length, comm)

Performs halo communication for a one-dimensional array or subarray. Halo cells are updated with values from neighboring processors.

# Arguments
- `u`: An array or subarray that represents the primary dataset.
- `halo`: An array or subarray that represents the halo cells.
- `send_id`: An array of IDs for the processors to which data will be sent.
- `recv_id`: An array of IDs for the processors from which data will be received.
- `send_idx`: An array of indices in `u` that mark the data to be sent.
- `recv_length`: The number of elements to be received from each processor.
- `comm`: The MPI communicator.

# Returns
- `u`: The updated primary dataset.
- `halo`: The updated halo cells.
"""
function perform_halo_update!(u::Union{Array{Float64, 1}, SubArray{Float64, 1}},
                              halo::Union{Array{Float64, 1}, SubArray{Float64, 1}}, send_id,
                              recv_id, send_idx, recv_length, comm)
    # Perform halo communication of 1D halo
    # println("Performing 1D halo communication...")
    displs = [0; cumsum(recv_length)[1:(end - 1)]]
    reqs = MPI.Request[]
    for i in eachindex(send_id)
        send_halo = u[send_idx[i]]
        recv_halo = @view(halo[(displs[i] + 1):(displs[i] + recv_length[i])])
        rreq = MPI.Irecv!(recv_halo, comm; source = recv_id[i] - 1, tag = 0)
        sreq = MPI.Isend(send_halo, comm; dest = send_id[i] - 1, tag = 0)
        push!(reqs, rreq)
        push!(reqs, sreq)
    end
    MPI.Waitall!(reqs)

    return u, halo
end

"""
    perform_halo_update!(u::Union{Array{Float64,2},SubArray{Float64,2}}, halo::Union{Array{Float64,2},SubArray{Float64,2}}, send_id, recv_id, send_idx, recv_length, comm)

Performs halo communication for a two-dimensional array or subarray. Halo cells are updated with values from neighboring processors.

# Arguments
- `u`: An array or subarray that represents the primary dataset.
- `halo`: An array or subarray that represents the halo cells.
- `send_id`: An array of IDs for the processors to which data will be sent.
- `recv_id`: An array of IDs for the processors from which data will be received.
- `send_idx`: An array of indices in `u` that mark the data to be sent.
- `recv_length`: The number of elements to be received from each processor.
- `comm`: The MPI communicator.

# Returns
- `u`: The updated primary dataset.
- `halo`: The updated halo cells.
"""
function perform_halo_update!(u::Union{Array{Float64, 2}, SubArray{Float64, 2}},
                              halo::Union{Array{Float64, 2}, SubArray{Float64, 2}}, send_id,
                              recv_id, send_idx, recv_length, comm)
    # process the 2-dimensional view here
    # println("Performing 2D halo communication...")
    displs = [0; cumsum(recv_length)[1:(end - 1)]]
    reqs = MPI.Request[]
    for i in eachindex(send_id)
        send_halo = u[send_idx[i], :]
        recv_halo = @view(halo[(displs[i] + 1):(displs[i] + recv_length[i]), :])
        # for j in 0:MPI.Comm_size(comm)-1
        #     if MPI.Comm_rank(comm) == j
        #         println("Rank: ", MPI.Comm_rank(comm))
        #         println("send_halo: ", Base.size(send_halo))
        #         println("recv_halo: ", Base.size(recv_halo))
        #         println("recv_length: ", recv_length[i])
        #     end
        # end
        rreq = MPI.Irecv!(recv_halo, comm; source = recv_id[i] - 1, tag = 0)
        sreq = MPI.Isend(send_halo, comm; dest = send_id[i] - 1, tag = 0)
        push!(reqs, rreq)
        push!(reqs, sreq)
    end
    MPI.Waitall!(reqs)

    return u, halo
end

"""
    perform_halo_update!(u::StructArray{SVector{N, T}, 2, NTuple{N, SubArray{T, 2}}}, 
                         halo::StructArray{SVector{N, T}, 2, NTuple{N, SubArray{T, 2}}}, 
                         send_id, recv_id, send_idx, recv_length, comm)

Performs halo communication for a two-dimensional StructArray of SVectors of arbitrary size. 
Halo cells are updated with values from neighboring processors.

# Arguments
- `u`: StructArray representing the primary dataset.
- `halo`: StructArray representing the halo cells.
- `send_id`: Array of IDs for the processors to which data will be sent.
- `recv_id`: Array of IDs for the processors from which data will be received.
- `send_idx`: Array of indices in `u` marking the data to be sent.
- `recv_length`: Number of elements to be received from each processor.
- `comm`: MPI communicator.

# Returns
- `u`: The updated primary dataset.
- `halo`: The updated halo cells.
"""
function perform_halo_update!(u::T, halo::T,
                              send_id::Vector{Int}, recv_id::Vector{Int},
                              send_idx::Vector{Vector{Int}}, recv_length::Vector{Int},
                              mpi_cache) where {T <:
                                                Union{StructArray{<:SVector},
                                                      StructVector{<:SVector}}}
    comm = mpi_cache.comm
    mpi_send_buffers = mpi_cache.mpi_send_buffers
    mpi_recv_buffers = mpi_cache.mpi_recv_buffers
    displs = [0; cumsum(recv_length)[1:(end - 1)]]
    reqs = MPI.Request[]
    println("Rank ", MPI.Comm_rank(comm), " performing halo update...")
    for i in eachindex(send_id)
        send_buffer = mpi_send_buffers[i]
        recv_buffer = mpi_recv_buffers[i]
        send_halo_view = @view u[send_idx[i]]
        recv_halo_view = @view halo[(displs[i] + 1):(displs[i] + recv_length[i])]
        for j in 1:length(send_buffer)
            send_buffer[j] = send_halo_view[j]
        end
        # send_buffer .= send_halo_view
        # recv_buffer .= recv_halo_view
        rreq = MPI.Irecv!(recv_buffer, comm; source = recv_id[i] - 1, tag = 0)
        sreq = MPI.Isend(send_buffer, comm; dest = send_id[i] - 1, tag = 0)
        for j in 1:length(recv_buffer)
            recv_halo_view[j] = recv_buffer[j]
        end
        push!(reqs, rreq)
        push!(reqs, sreq)
        # if MPI.Comm_rank(comm) == 0
        #     println("Rank: ", MPI.Comm_rank(comm), " send to ", send_id[i] - 1, " recv from ", recv_id[i] - 1)
        # end
        println("Rank: ", MPI.Comm_rank(comm), " send to ", send_id[i] - 1, " recv from ",
                recv_id[i] - 1)
    end
    MPI.Waitall!(reqs)
    println("Halo update complete on rank: ", MPI.Comm_rank(comm))

    return u, halo
end
