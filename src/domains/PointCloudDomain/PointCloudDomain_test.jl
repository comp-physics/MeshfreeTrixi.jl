# using Revise
# using StaticArrays
# using ConstructionBase
# includet("PointCloudDomain.jl")

# Example boundary tags with indices and normals
boundary_tags = Dict(:top => BoundaryData([1, 2, 3],
                                          [
                                              SVector{3}(0.0, 1.0, 0.0),
                                              SVector{3}(0.0, 1.0, 0.0),
                                              SVector{3}(0.0, 1.0, 0.0)
                                          ]),
                     :side => BoundaryData([4, 5],
                                           [
                                               SVector{3}(1.0, 0.0, 0.0),
                                               SVector{3}(1.0, 0.0, 0.0)
                                           ]))

# Example initialization of PointCloudMeshData
points = [SVector{3}(rand(3)) for _ in 1:5] # Sample points
neighbors = [rand(1:5, 3) for _ in 1:5] # Sample connectivity/neighbors
domain = PointCloudDomain(points, neighbors, boundary_tags)

# Example boundary tags with indices and normals
boundary_tags = Dict(:top => BoundaryData([1, 2, 3],
                                          [
                                              SVector{2}(0.0, 1.0),
                                              SVector{2}(0.0, 1.0),
                                              SVector{2}(0.0, 1.0)
                                          ]),
                     :side => BoundaryData([4, 5],
                                           [SVector{2}(1.0, 0.0), SVector{2}(1.0, 0.0)]))

# Example initialization of PointCloudMeshData
points = [SVector{2}(rand(2)) for _ in 1:5] # Sample points
neighbors = [rand(1:5, 6) for _ in 1:5] # Sample connectivity/neighbors
domain = PointCloudDomain(points, neighbors, boundary_tags)
