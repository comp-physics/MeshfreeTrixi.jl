# Similar to StartUpDG.MeshData
# This is the underlying data structure for PointCloudDomain
# Mostly reworked 
struct PointData{Dim, Tv, Ti}
    points::Vector{Tv}                # Point coordinates
    neighbors::Vector{Vector{Ti}}     # Neighbors for each point
    num_points::Int                   # Number of points
    num_neighbors::Int                # Number of neighbors lists
    dx_min::Float64                   # Minimum distance between points
    dx_avg::Float64                   # Average distance between points
end

## We need to calculate dx_min at runtime so we cannot use this constructor
# function PointData(points::Vector{Tv},
#                    neighbors::Vector{Vector{Ti}}) where {
#                                                          Dim,
#                                                          Tv <: SVector{Dim, Float64},
#                                                          Ti
#                                                          }
#     PointData{Dim, Tv, Ti}(points, neighbors, length(points), length(neighbors[1]))
# end

"""
    PointData(medusa_data::Vector{Tv}, basis) where {Tv <: SVector{Dim, Float64}}

- `medusa_data` contains point positions for entire point cloud.
- `basis` contains all basis information including the number of required neighbors to support required order of accuracy.
"""
function PointData(medusa_data::Vector{Tv},
                   basis::RefPointData) where {Dim, Tv <: SVector{Dim, Float64}}
    nv = basis.nv  # The number of neighbors

    # Calculate neighbor list for all points
    kdtree = KDTree(medusa_data)
    n_idxs, n_dists = knn(kdtree, medusa_data, nv, true)
    dx_min = minimum(n_dists)

    # Calculate avg and min distances between points
    idxs_y, dists_y = knn(kdtree, medusa_data, 2, true)
    dx_avg = mean(dists_y)[2]
    dx_min = minimum(dists_y)[2]

    # Instantiate PointData with the points and neighbors. The num_points and num_neighbors are automatically computed.
    return PointData{Dim, Tv, Int}(medusa_data, n_idxs, length(medusa_data), nv, dx_min,
                                   dx_avg)
end

struct BoundaryData{Ti <: Integer, Tv <: SVector{N, T} where {N, T <: Number}}
    idx::Vector{Ti}       # Indices of boundary points
    normals::Vector{Tv}   # Normals at boundary points
end

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

Base.ndims(::PointCloudDomain{NDIMS}) where {NDIMS} = NDIMS

# function Base.show(io::IO,
#                    mesh::PointCloudDomain{NDIMS, MeshType}) where {NDIMS, MeshType}
#     @nospecialize mesh # reduce precompilation time
#     print(io, "$MeshType PointCloudDomain with NDIMS = $NDIMS.")
# end

function Base.show(io::IO, mesh::PointCloudDomain{Dim, Tv, Ti}) where {Dim, Tv, Ti}
    print(io, "PointCloudDomain with NDIMS = $Dim")
end

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

function Base.show(io::IO, point_data::PointData{Dim, Tv, Ti}) where {Dim, Tv, Ti}
    print(io, "PointData with NDIMS = $Dim")
end

function Base.show(io::IO, ::MIME"text/plain",
                   point_data::PointData{Dim, Tv, Ti}) where {Dim, Tv, Ti}
    if get(io, :compact, false)
        show(io, point_data)
    else
        num_points_to_show = 5

        summary_header(io, "PointData{$Dim}")
        summary_line(io, "Number of points", point_data.num_points)
        summary_line(io, "Vector type of points", "$(eltype(point_data.points))")
        summary_line(io, "Number of neighbors per point", point_data.num_neighbors)

        # Show a sample of points
        # summary_line(io, "First points", "")
        for i in 1:min(num_points_to_show, point_data.num_points)
            summary_line(increment_indent(io), "Point $i", "$(point_data.points[i])")
        end
        if point_data.num_points > num_points_to_show
            summary_line(increment_indent(io), "...", "...")
            # Optionally, show a few points from the end if the list is long
            for i in (point_data.num_points - num_points_to_show + 1):(point_data.num_points)
                summary_line(increment_indent(io), "Point $i", "$(point_data.points[i])")
            end
        end

        # Optionally, include similar logic for showing a sample of neighbor lists if desired

        summary_footer(io)
    end
end

function Base.show(io::IO, boundary_data::BoundaryData{Ti, Tv}) where {Ti, Tv}
    print(io, "BoundaryData with index type = $Ti, normal type = $Tv")
end

function Base.show(io::IO, ::MIME"text/plain",
                   boundary_data::BoundaryData{Ti, Tv}) where {Ti, Tv}
    if get(io, :compact, false)
        show(io, boundary_data)
    else
        num_normals_to_show_start = 3
        num_normals_to_show_end = 2

        # Assuming you have a similar summary utilities as in the previous example
        summary_header(io, "BoundaryData")
        summary_line(io, "Number of boundary points", length(boundary_data.idx))
        summary_line(io, "Vector type of normals", "$(eltype(boundary_data.normals))")

        # Check how many normals there are
        num_normals = length(boundary_data.normals)
        if num_normals > num_normals_to_show_start + num_normals_to_show_end
            # Show the first few normals
            for i in 1:num_normals_to_show_start
                summary_line(increment_indent(io), "Normal $i",
                             "$(boundary_data.normals[i])")
            end
            summary_line(increment_indent(io), "Normal ...", "...")
            # Show the last few normals
            for i in (num_normals - num_normals_to_show_end + 1):num_normals
                summary_line(increment_indent(io), "Normal $i",
                             "$(boundary_data.normals[i])")
            end
        else
            # If there aren't many normals, just show all of them
            for i in 1:num_normals
                summary_line(io, "Normal $i", "$(boundary_data.normals[i])")
            end
        end

        summary_footer(io)
    end
end
