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

# function perform_halo_update!(u::Union{MPIArray{Float64, 2},
#                                        SubArray{T, N1, <:MPIArray{T, M1}}},
#                               halo::Union{MPIArray{Float64, 2},
#                                           SubArray{T, N2, <:MPIArray{T, M2}}},
#                               send_id, recv_id, send_idx, recv_length,
#                               comm) where {T, N1, M1, N2, M2}
#     # process the 2-dimensional view here
#     # println("Performing 2D halo communication...")
#     displs = [0; cumsum(recv_length)[1:(end - 1)]]
#     reqs = MPI.Request[]
#     for i in eachindex(send_id)
#         send_halo = u[send_idx[i], :]
#         recv_halo = @view(halo[(displs[i] + 1):(displs[i] + recv_length[i]), :])
#         # for j in 0:MPI.Comm_size(comm)-1
#         #     if MPI.Comm_rank(comm) == j
#         #         println("Rank: ", MPI.Comm_rank(comm))
#         #         println("send_halo: ", Base.size(send_halo))
#         #         println("recv_halo: ", Base.size(recv_halo))
#         #         println("recv_length: ", recv_length[i])
#         #     end
#         # end
#         rreq = MPI.Irecv!(recv_halo.data, comm; source = recv_id[i] - 1, tag = 0)
#         sreq = MPI.Isend(send_halo.data, comm; dest = send_id[i] - 1, tag = 0)
#         push!(reqs, rreq)
#         push!(reqs, sreq)
#     end
#     MPI.Waitall!(reqs)

#     return u, halo
# end
