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
struct SerialPointCloudDomain{NDIMS, PointDataT <: PointData{NDIMS}, BoundaryFaceT} <:
       PointCloudDomain{NDIMS}
    pd::PointDataT
    boundary_tags::BoundaryFaceT
    unsaved_changes::Bool # Required for SaveSolutionCallback
end

# Workaround so other calls to PointCloudDomain will still work
function SerialPointCloudDomain(pd::PointData{NDIMS, Tv, Ti},
                                boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {
                                                                                          NDIMS,
                                                                                          Tv,
                                                                                          Ti
                                                                                          }
    return SerialPointCloudDomain{NDIMS, PointData{NDIMS, Tv, Ti},
                                  Dict{Symbol, BoundaryData{Ti, Tv}}}(pd, boundary_tags,
                                                                      false)
end

function SerialPointCloudDomain(points::Vector{Tv}, neighbors::Vector{Vector{Ti}},
                                boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {
                                                                                          N,
                                                                                          Tv <:
                                                                                          SVector{N,
                                                                                                  Float64},
                                                                                          Ti
                                                                                          }
    pointData = PointData(points, neighbors)  # Create an instance of PointData
    return SerialPointCloudDomain(pointData,
                                  boundary_tags, false)
end

# Main function for instantiating all the necessary data for a SerialPointCloudDomain
function SerialPointCloudDomain(basis::RefPointData{NDIMS},
                                filename::String,
                                boundary_names_dict::Dict{Symbol, Int}) where {NDIMS}
    medusa_data, interior_idx, boundary_idxs, boundary_normals = read_medusa_file(filename)
    pd = PointData(medusa_data, basis)
    boundary_tags = Dict(name => BoundaryData(boundary_idxs[idx], boundary_normals[idx])
                         for (name, idx) in boundary_names_dict)
    return SerialPointCloudDomain(pd,
                                  boundary_tags, false)
end

Base.ndims(::PointCloudDomain{NDIMS}) where {NDIMS} = NDIMS
