# everything related to a DG semidiscretization in 2D using MPI,
# currently limited to Lobatto-Legendre nodes

# TODO: MPI dimension agnostic
# For u_local, halo = halo_update!(u_local, halo, send_id, recv_id, send_idx, recv_length, comm)
mutable struct MPICache{uEltype <: Real}
    # mpi_neighbor_ranks::Vector{Int}
    mpi_send_id::Vector{Int}
    mpi_recv_id::Vector{Int}
    halo_send_idx::Vector{Vector{Int}}
    halo_recv_length::Vector{Int}
    # mpi_neighbor_interfaces::Vector{Vector{Int}}
    # mpi_neighbor_mortars::Vector{Vector{Int}}
    mpi_send_buffers::Vector{Vector{uEltype}}
    mpi_recv_buffers::Vector{Vector{uEltype}}
    # mpi_send_requests::Vector{MPI.Request}
    # mpi_recv_requests::Vector{MPI.Request}
    mpi_requests::Vector{MPI.Request}
    # n_elements_by_rank::OffsetArray{Int, 1, Array{Int, 1}}
    n_elements_local::Int
    n_elements_global::Int
    # first_element_global_id::Int
end

function MPICache(uEltype, mpi_send_id::Vector{Int}, mpi_recv_id::Vector{Int},
                  halo_send_idx::Vector{Vector{Int}}, halo_recv_length::Vector{Int},
                  n_elements_local::Int, n_elements_global::Int)
    # MPI communication "just works" for bitstypes only
    if !isbitstype(uEltype)
        throw(ArgumentError("MPICache only supports bitstypes, $uEltype is not a bitstype."))
    end
    # mpi_neighbor_ranks = Vector{Int}(undef, 0)
    # mpi_send_id = Vector{Int}(undef, 0)
    # mpi_recv_id = Vector{Int}(undef, 0)
    # halo_send_idx = Vector{Vector{Int}}(undef, 0)
    # halo_recv_length = Vector{Int}(undef, 0)
    # mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, 0)
    # mpi_neighbor_mortars = Vector{Vector{Int}}(undef, 0)
    mpi_send_buffers = Vector{Vector{uEltype}}(undef, 0)
    mpi_recv_buffers = Vector{Vector{uEltype}}(undef, 0)
    # mpi_send_requests = Vector{MPI.Request}(undef, 0)
    # mpi_recv_requests = Vector{MPI.Request}(undef, 0)
    mpi_requests = Vector{MPI.Request}(undef, 0)
    # n_elements_by_rank = OffsetArray(Vector{Int}(undef, 0), 0:-1)
    # n_elements_local = 0
    # n_elements_global = 0
    # first_element_global_id = 0

    MPICache{uEltype}(mpi_send_id, mpi_recv_id, halo_send_idx, halo_recv_length,
                      mpi_send_buffers, mpi_recv_buffers, mpi_requests,
                      n_elements_local, n_elements_global)
end
@inline Base.eltype(::MPICache{uEltype}) where {uEltype} = uEltype

"""
    ParallelPointCloudDomain(NDIMS, PointDataT <: PointData{NDIMS}, BoundaryFaceT}

- `pd` contains point data structure.
- `boundary tags` dictionary of all boundary tags and associated point indices.
"""
# struct ParallelPointCloudDomain{Dim, Tv, Ti}
#     pd::PointData{Dim, Tv, Ti}  # Encapsulates points and neighbors
#     boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}  # Boundary data
# end
### Actual ParallelPointCloudDomain for dispatching problems with 
struct ParallelPointCloudDomain{NDIMS, PointDataT <: PointData{NDIMS}, BoundaryFaceT}
    pd::PointDataT
    boundary_tags::BoundaryFaceT
    mpi::MPICache
    unsaved_changes::Bool # Required for SaveSolutionCallback
end

# Workaround so other calls to ParallelPointCloudDomain will still work
function ParallelPointCloudDomain(pd::PointData{NDIMS, Tv, Ti},
                                  boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {
                                                                                            NDIMS,
                                                                                            Tv,
                                                                                            Ti
                                                                                            }
    return ParallelPointCloudDomain{NDIMS, PointData{NDIMS, Tv, Ti},
                                    Dict{Symbol, BoundaryData{Ti, Tv}}}(pd, boundary_tags,
                                                                        false)
end

function ParallelPointCloudDomain(points::Vector{Tv}, neighbors::Vector{Vector{Ti}},
                                  boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {
                                                                                            N,
                                                                                            Tv <:
                                                                                            SVector{N,
                                                                                                    Float64},
                                                                                            Ti
                                                                                            }
    pointData = PointData(points, neighbors)  # Create an instance of PointData
    return ParallelPointCloudDomain(pointData,
                                    boundary_tags, false)
end

# Main function for instantiating all the necessary data for a ParallelPointCloudDomain
function ParallelPointCloudDomain(basis::RefPointData{NDIMS},
                                  filename::String,
                                  boundary_names_dict::Dict{Symbol, Int}) where {NDIMS}
    # medusa_data, interior_idx, boundary_idxs, boundary_normals = read_medusa_file(filename)

    # Pre-allocate MPI cache. Currently rank 0 loads all data and 
    # distributes to other ranks.
    # Pre-process everything then set up our local MPI cache
    # followed by the rest of the data structure.
    num_procs = mpi_nranks()
    local_points, local_to_global_idx,
    halo_points, halo_global, halo_proc, halo_global_to_local_idx,
    boundary_global, boundary_normals_local, boundary_local_idxs,
    boundary_halo_global, boundary_normals_halo, boundary_halo_idxs,
    send_id, recv_id, send_idx, recv_length, dx_min, dx_avg = preprocess(filename,
                                                                         2 * basis.nv,
                                                                         num_procs)

    num_local_points = length(local_points)
    num_halo_points = length(halo_points)

    mpi_cache = MPICache(real(basis), send_id, recv_id, send_idx, recv_length,
                         num_local_points, num_glocal_points)

    points = vcat(local_points, halo_points)
    pd = PointData(points, solver.basis, dx_min, dx_avg)

    # Combine boundary data
    boundary_idxs = deepcopy(boundary_local_idxs)
    boundary_normals = deepcopy(boundary_normals_local)
    for i in eachindex(boundary_idxs)
        append!(boundary_idxs[i], boundary_halo_idxs[i])
        append!(boundary_normals[i], boundary_normals_halo[i])
    end
    boundary_tags = Dict(name => BoundaryData(boundary_idxs[idx], boundary_normals[idx])
                         for (name, idx) in boundary_names_dict)
    return ParallelPointCloudDomain(pd,
                                    boundary_tags, mpi_cache, false)

    # return ParallelPointCloudDomain{NDIMS, PointData{NDIMS, Tv, Ti},
    #                                 Dict{Symbol, BoundaryData{Ti, Tv}}}(pd, boundary_tags,
    #                                                                     mpi, false)
end

Base.ndims(::ParallelPointCloudDomain{NDIMS}) where {NDIMS} = NDIMS
