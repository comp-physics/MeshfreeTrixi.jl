# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    PointCloudDomain{NDIMS, ...}

`PointCloudDomain` describes a domain type which wraps `StartUpDG.MeshData` and `boundary_faces` in a
dispatchable type. This is intended to store geometric data and connectivities for any type of
mesh (Cartesian, affine, curved, structured/unstructured).
"""
### Basing Medusa style PointCloudDomain on DGMultiMesh
# Planning to have generic PointCloudDomain but a specific constructor 
# for Medusa input files. Eventually could call Medusa directly.
struct PointCloudDomain{NDIMS, MeshType, MeshDataT <: MeshData{NDIMS}, BoundaryFaceT}
    md::MeshDataT
    boundary_faces::BoundaryFaceT
end

# enable use of @set and setproperties(...) for PointCloudDomain
function ConstructionBase.constructorof(::Type{PointCloudDomain{T1, T2, T3, T4}}) where {
                                                                                         T1,
                                                                                         T2,
                                                                                         T3,
                                                                                         T4
                                                                                         }
    PointCloudDomain{T1, T2, T3, T4}
end

Base.ndims(::PointCloudDomain{NDIMS}) where {NDIMS} = NDIMS

function Base.show(io::IO,
                   mesh::PointCloudDomain{NDIMS, MeshType}) where {NDIMS, MeshType}
    @nospecialize mesh # reduce precompilation time
    print(io, "$MeshType PointCloudDomain with NDIMS = $NDIMS.")
end

function Base.show(io::IO, ::MIME"text/plain",
                   mesh::PointCloudDomain{NDIMS, MeshType}) where {NDIMS, MeshType}
    @nospecialize mesh # reduce precompilation time
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io, "PointCloudDomain{$NDIMS, $MeshType}, ")
        summary_line(io, "number of elements", mesh.md.num_elements)
        summary_line(io, "number of boundaries", length(mesh.boundary_faces))
        for (boundary_name, faces) in mesh.boundary_faces
            summary_line(increment_indent(io), "nfaces on $boundary_name",
                         length(faces))
        end
        summary_footer(io)
    end
end

# Additional Parser Methods Below. 

# MeshData comes from StartUpDG.MeshData. Boundary's are a dictionary of boundary names and
# come from StartUpDG.tag_boundary_faces.
# Furthermore, this mesh type is generic. The constructor is called by the solver using 
# another call that has the appropriate solver::type. 

# The way our domain data will be instantiated is as follows.
# Generate solver with polydeg and specific basis.
#    solver = PointCloudSolver{polydeg = 3, approximation_type = PHS_w_Monomials()}
# Generate mesh where solver is an input. This calls the mesh constructor in solver file.
#    mesh = PointCloudMesh(filename, solver)
# This allows us to calculate the NearestNeighbors connectivity and store it in the domain.
# This is because we will already know the dimensionality and polydeg. 
# Within mesh constructor in PointCloudSolver FILE we will also use part of the basis 
# information in order to pull in element type and dimensionality. 

end # @muladd
