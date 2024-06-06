# everything related to a DG semidiscretization in 2D using MPI,
# currently limited to Lobatto-Legendre nodes

# TODO: MPI dimension agnostic
mutable struct MPICache{uEltype <: Real}
    mpi_neighbor_ranks::Vector{Int}
    mpi_neighbor_interfaces::Vector{Vector{Int}}
    mpi_neighbor_mortars::Vector{Vector{Int}}
    mpi_send_buffers::Vector{Vector{uEltype}}
    mpi_recv_buffers::Vector{Vector{uEltype}}
    mpi_send_requests::Vector{MPI.Request}
    mpi_recv_requests::Vector{MPI.Request}
    n_elements_by_rank::OffsetArray{Int, 1, Array{Int, 1}}
    n_elements_global::Int
    first_element_global_id::Int
end

function MPICache(uEltype)
    # MPI communication "just works" for bitstypes only
    if !isbitstype(uEltype)
        throw(ArgumentError("MPICache only supports bitstypes, $uEltype is not a bitstype."))
    end
    mpi_neighbor_ranks = Vector{Int}(undef, 0)
    mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, 0)
    mpi_neighbor_mortars = Vector{Vector{Int}}(undef, 0)
    mpi_send_buffers = Vector{Vector{uEltype}}(undef, 0)
    mpi_recv_buffers = Vector{Vector{uEltype}}(undef, 0)
    mpi_send_requests = Vector{MPI.Request}(undef, 0)
    mpi_recv_requests = Vector{MPI.Request}(undef, 0)
    n_elements_by_rank = OffsetArray(Vector{Int}(undef, 0), 0:-1)
    n_elements_global = 0
    first_element_global_id = 0

    MPICache{uEltype}(mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars,
                      mpi_send_buffers, mpi_recv_buffers,
                      mpi_send_requests, mpi_recv_requests,
                      n_elements_by_rank, n_elements_global,
                      first_element_global_id)
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
                                  points::Vector{SVector{NDIMS, Float64}},
                                  boundary_idxs::Vector{Vector{Int}},
                                  boundary_normals::Vector{Vector{SVector{NDIMS, Float64}}},
                                  boundary_names_dict::Dict{Symbol, Int}) where {NDIMS}
    # Update to construct MPI cache
    medusa_data, interior_idx, boundary_idxs, boundary_normals = read_medusa_file(filename)
    pd = PointData(medusa_data, solver.basis)
    boundary_tags = Dict(name => BoundaryData(boundary_idxs[idx], boundary_normals[idx])
                         for (name, idx) in boundary_names_dict)
    return ParallelPointCloudDomain(pd,
                                    boundary_tags, mpi, false)

    # return ParallelPointCloudDomain{NDIMS, PointData{NDIMS, Tv, Ti},
    #                                 Dict{Symbol, BoundaryData{Ti, Tv}}}(pd, boundary_tags,
    #                                                                     mpi, false)
end

Base.ndims(::ParallelPointCloudDomain{NDIMS}) where {NDIMS} = NDIMS
