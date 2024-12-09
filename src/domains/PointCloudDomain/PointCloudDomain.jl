"""
    PointCloudDomain(NDIMS, PointDataT <: PointData{NDIMS}, BoundaryFaceT}

Domain specification containing point cloud data structure and boundary tags 
for a point cloud domain. Includes specialization for MPI parallelism.
"""
abstract type PointCloudDomain{NDIMS} end

include("SerialPointCloud.jl")
# MPI-based PointCloudDomain
include("scatter_pointcloud.jl")
include("partition_domain.jl")
include("ParallelPointCloud.jl")

# const SerialPointCloudDomain{NDIMS} = PointCloudDomain{NDIMS, <:SerialPointCloud{NDIMS}}
# const ParallelPointCloudDomain{NDIMS} = PointCloudDomain{NDIMS, <:ParallelPointCloud{NDIMS}}

@inline mpi_parallel(mesh::PointCloudDomain) = False()
# @inline mpi_parallel(mesh::SerialPointCloudDomain) = False()
@inline mpi_parallel(mesh::ParallelPointCloudDomain) = True()

# Primary constructor for MPI-aware PointCloudDomain
function PointCloudDomain(basis::RefPointData{NDIMS},
                          filename::String,
                          boundary_names_dict::Dict{Symbol, Int}) where {NDIMS}

    # TODO: MPI, create nice interface for a parallel tree/mesh
    if mpi_isparallel()
        # TreeType = ParallelTree{NDIMS}
        return ParallelPointCloudDomain(basis, filename,
                                        boundary_names_dict)
    else
        # TreeType = SerialTree{NDIMS}
        return SerialPointCloudDomain(basis, filename,
                                      boundary_names_dict)
    end

    # Create mesh
    # mesh = @trixi_timeit timer() "creation" TreeMesh{NDIMS, TreeType}(n_cells_max,
    #                                                                   domain_center,
    #                                                                   domain_length,
    #                                                                   periodicity)

    # # Initialize mesh
    # initialize!(mesh, initial_refinement_level, refinement_patches, coarsening_patches)

    # return mesh
end

# function Base.show(io::IO,
#                    mesh::PointCloudDomain{NDIMS, MeshType}) where {NDIMS, MeshType}
#     @nospecialize mesh # reduce precompilation time
#     print(io, "$MeshType PointCloudDomain with NDIMS = $NDIMS.")
# end

function Base.show(io::IO,
                   mesh::Union{SerialPointCloudDomain{Dim, Tv, Ti},
                               ParallelPointCloudDomain{Dim, Tv, Ti}}) where {Dim, Tv, Ti}
    print(io, "PointCloudDomain with NDIMS = $Dim")
end

function Base.show(io::IO, ::MIME"text/plain",
                   mesh::Union{SerialPointCloudDomain{Dim, Tv, Ti},
                               ParallelPointCloudDomain{Dim, Tv, Ti}}) where {Dim, Tv, Ti}
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
