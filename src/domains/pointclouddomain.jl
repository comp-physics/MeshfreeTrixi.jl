using Revise
using StaticArrays
using Trixi: summary_header, summary_line, summary_footer, increment_indent

includet("geometry_primatives.jl")

struct PointData{Tv, Ti}
    points::Vector{Tv}                # Point coordinates
    neighbors::Vector{Vector{Ti}}     # Neighbors for each point
    num_points::Int                   # Number of points
    num_neighbors::Int                # Number of neighbors lists

    function PointData{Tv, Ti}(points::Vector{Tv},
                               neighbors::Vector{Vector{Ti}}) where {Tv, Ti}
        new{Tv, Ti}(points, neighbors, length(points), length(neighbors[1]))
    end
end

struct BoundaryData{Ti <: Integer, Tv <: SVector{N, T} where {N, T <: Number}}
    idx::Vector{Ti}       # Indices of boundary points
    normals::Vector{Tv}   # Normals at boundary points
end

struct PointCloudDomain{Dim, Tv, Ti}
    pd::PointData{Tv, Ti}  # Encapsulates points and neighbors
    boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}  # Boundary data
end

function PointCloudDomain(points::Vector{Tv}, neighbors::Vector{Vector{Ti}},
                          boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {N,
                                                                                    Tv <:
                                                                                    SVector{
                                                                                            N,
                                                                                            Float64
                                                                                            },
                                                                                    Ti}
    pointData = PointData{Tv, Ti}(points, neighbors)  # Create an instance of PointData
    return PointCloudDomain{N, Tv, Ti}(pointData, boundary_tags)
end

# Base.ndims(::PointCloudDomain{NDIMS}) where {NDIMS} = NDIMS

# function Base.show(io::IO,
#                    mesh::PointCloudDomain{NDIMS, MeshType}) where {NDIMS, MeshType}
#     @nospecialize mesh # reduce precompilation time
#     print(io, "$MeshType PointCloudDomain with NDIMS = $NDIMS.")
# end

function Base.show(io::IO, mesh::PointCloudDomain{Dim, Tv, Ti}) where {Dim, Tv, Ti}
    print(io, "PointCloudDomain with dimension = $Dim, point type = $Tv, index type = $Ti")
end

# function Base.show(io::IO, ::MIME"text/plain",
#                    mesh::PointCloudDomain{NDIMS, MeshType}) where {NDIMS, MeshType}
#     @nospecialize mesh # reduce precompilation time
#     if get(io, :compact, false)
#         show(io, mesh)
#     else
#         summary_header(io, "PointCloudDomain{$NDIMS, $MeshType}, ")
#         summary_line(io, "number of elements", mesh.md.num_elements)
#         summary_line(io, "number of boundaries", length(mesh.boundary_faces))
#         for (boundary_name, faces) in mesh.boundary_faces
#             summary_line(increment_indent(io), "nfaces on $boundary_name",
#                          length(faces))
#         end
#         summary_footer(io)
#     end
# end

function Base.show(io::IO, ::MIME"text/plain",
                   mesh::PointCloudDomain{Dim, Tv, Ti}) where {Dim, Tv, Ti}
    if get(io, :compact, false)
        show(io, mesh)
    else
        # Use Trixi's summary functions for structured output
        summary_header(io, "PointCloudDomain{$Dim}")
        summary_line(io, "Number of points", mesh.pd.num_points)
        summary_line(io, "Number of neighbors", mesh.pd.num_neighbors)

        # For boundary tags, we'll handle them slightly differently to include both count and detail
        boundary_tags_count = length(mesh.boundary_tags)
        summary_line(io, "Number of boundary tags", boundary_tags_count)

        for (boundary_name, data) in mesh.boundary_tags
            boundary_points_count = length(data.idx)
            summary_line(increment_indent(io), "Boundary '$boundary_name'",
                         "$boundary_points_count boundary points")
        end

        summary_footer(io)
    end
end
