"""
    PointCloudDomain(NDIMS, PointDataT <: PointData{NDIMS}, BoundaryFaceT}

- `pd` contains point data structure.
- `boundary tags` dictionary of all boundary tags and associated point indices.
"""
# struct PointCloudDomain{Dim, Tv, Ti}
#     pd::PointData{Dim, Tv, Ti}  # Encapsulates points and neighbors
#     boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}  # Boundary data
# end
### Actual PointCloudDomain for dispatching problems with 
struct PointCloudDomain{NDIMS, PointDataT <: PointData{NDIMS}, BoundaryFaceT}
    pd::PointDataT
    boundary_tags::BoundaryFaceT
    unsaved_changes::Bool # Required for SaveSolutionCallback
end

# Workaround so other calls to PointCloudDomain will still work
function PointCloudDomain(pd::PointData{NDIMS, Tv, Ti},
                          boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {NDIMS,
                                                                                    Tv, Ti}
    return PointCloudDomain{NDIMS, PointData{NDIMS, Tv, Ti},
                            Dict{Symbol, BoundaryData{Ti, Tv}}}(pd, boundary_tags, false)
end

function PointCloudDomain(points::Vector{Tv}, neighbors::Vector{Vector{Ti}},
                          boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {
                                                                                    N,
                                                                                    Tv <:
                                                                                    SVector{N,
                                                                                            Float64},
                                                                                    Ti
                                                                                    }
    pointData = PointData(points, neighbors)  # Create an instance of PointData
    return PointCloudDomain(pointData,
                            boundary_tags, false)
end

# Main function for instantiating all the necessary data for a SerialPointCloudDomain
function SerialPointCloudDomain(basis::RefPointData{NDIMS},
                                points::Vector{SVector{NDIMS, Float64}},
                                boundary_idxs::Vector{Vector{Int}},
                                boundary_normals::Vector{Vector{SVector{NDIMS,
                                                                        Float64}}},
                                boundary_names_dict::Dict{Symbol, Int}) where {NDIMS}
    medusa_data, interior_idx, boundary_idxs, boundary_normals = read_medusa_file(filename)
    pd = PointData(medusa_data, solver.basis)
    boundary_tags = Dict(name => BoundaryData(boundary_idxs[idx], boundary_normals[idx])
                         for (name, idx) in boundary_names_dict)
    return PointCloudDomain(pd,
                            boundary_tags, false)
end

Base.ndims(::PointCloudDomain{NDIMS}) where {NDIMS} = NDIMS
