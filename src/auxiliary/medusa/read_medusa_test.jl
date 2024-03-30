using Revise

## Read in .txt
using DelimitedFiles
using GLMakie
includet("./import_medusa.jl")
includet("../../PointCloudDomain/geometry_primatives.jl")
includet("../../PointCloudDomain/PointCloudDomain.jl")

casename = "./medusa_point_clouds/cyl";
positions, interior_idx, boundary_idxs, boundary_normals, normals, boundary_ranges, types = import_medusa(casename)

# interior = positions[interior_idx, :]
scatter(Tuple.(positions), color = :black, markersize = 10.0, marker = :circle,
        axis = (aspect = DataAspect(),))

## Plot boundaries
for i in eachindex(boundary_idxs)
    boundary = Tuple.(positions[boundary_idxs[i]])
    scatter!(boundary, markersize = 15.0, marker = :circle)
end

# Example boundary tags with indices and normals
boundary_tags = Dict(:inlet => BoundaryData(boundary_idxs[1],
                                            boundary_normals[1]),
                     :outlet => BoundaryData(boundary_idxs[2],
                                             boundary_normals[2]),
                     :bottom => BoundaryData(boundary_idxs[3],
                                             boundary_normals[3]),
                     :top => BoundaryData(boundary_idxs[4],
                                          boundary_normals[4]),
                     :side => BoundaryData(boundary_idxs[5],
                                           boundary_normals[5]))

# We need the basis information so we can determine the number of neighbors! 

# Example initialization of PointCloudMeshData
# points = [SVector{3}(rand(3)) for _ in 1:5] # Sample points
neighbors = [rand(1:5, 3) for _ in 1:5] # Sample connectivity/neighbors
domain = PointCloudDomain(points, neighbors, boundary_tags)