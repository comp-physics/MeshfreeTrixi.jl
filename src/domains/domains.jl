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

include("pointcloud_domain.jl")
# include("structured_mesh.jl")
# include("surface_interpolant.jl")
# include("unstructured_mesh.jl")
# include("face_interpolant.jl")
# include("transfinite_mappings_3d.jl")
# include("p4est_mesh.jl")
# include("t8code_mesh.jl")
include("mesh_io.jl")
# include("dgmulti_meshes.jl")
end # @muladd