####################################################################################################
# Include files with actual implementations for different types of meshfree discretizations

# include("point_domain.jl")
# include("cell_domain.jl")

# include("distributed_point_domain.jl")
# include("distributed_cell_domain.jl")
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("PointCloudDomain/geometry_primatives.jl")
include("PointCloudDomain/PointCloudDomain.jl")
end # @muladd
