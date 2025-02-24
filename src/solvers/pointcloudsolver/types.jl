# Based on Trixi/solvers/DGMulti/types.jl
# Rewritten to work with RBF-FD methods on PointCloudDomains.
# Note: we define type aliases outside of the @muladd block to avoid Revise breaking when code
# inside the @muladd block is edited. See https://github.com/trixi-framework/Trixi.jl/issues/801
# for more details.

# `PointCloudSolver` refers to both multiple RBFSolver types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
const PointCloudSolver{NDIMS, ElemType, ApproxType, Engine} = RBFSolver{<:RefPointData{NDIMS,
                                                                                       ElemType,
                                                                                       ApproxType},
                                                                        Engine} where {
                                                                                       Engine
                                                                                       }

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# these are necessary for pretty printing
polydeg(solver::PointCloudSolver) = solver.basis.N
# function Base.summary(io::IO, solver::RBFSolver) where {RBFSolver <: PointCloudSolver}
#     print(io, "PointCloudSolver(polydeg=$(polydeg(solver)))")
# end

# real(rd) is the eltype of the nodes `rd.r`.
# Base.real(rd::RefPointData) = eltype(rd.r)
Base.real(rd::RefPointData) = Float64
# Currently RefPointData does not have eltype

"""
    PointCloudSolver(; polydeg::Integer,
              element_type::AbstractElemShape,
              approximation_type=RBF(),
              engine=RBFFDEngine(),
              RefElemData_kwargs...)

Create an RBF-FD method which uses
- approximations of polynomial degree `polydeg` using 'rbf_type' radial basis functions.
- element type `element_type` (`Point1D()`, `Point2D()`, and `Point3D()` currently supported)

Optional:
- `approximation_type` (default is `RBF()`).
- `RefElemData_kwargs` are additional keyword arguments for `RefPointData`.
"""
function PointCloudSolver(; polydeg = nothing,
                          element_type::AbstractElemShape,
                          approximation_type = RBF(),
                          engine = RBFFDEngine(),
                          kwargs...)

    # call dispatchable constructor
    PointCloudSolver(element_type, approximation_type, engine,
                     polydeg = polydeg, kwargs...)
end

# dispatchable constructor for PointCloudSolver to allow for specialization
function PointCloudSolver(element_type::AbstractElemShape,
                          approximation_type,
                          engine,
                          polydeg::Integer,
                          kwargs...)
    rd = RefPointData(element_type, approximation_type, polydeg; kwargs...)
    # `nothing` is passed as `mortar`
    return RBFSolver(rd, engine)
end

function PointCloudSolver(basis::RefPointData; engine = RBFFDEngine())
    # `nothing` is passed as `mortar`
    RBFSolver(basis, engine)
end

"""
    PointCloudBasis(element_type, polydeg; approximation_type = RBF(), kwargs...)

Constructs a basis for PointCloudSolver solvers. Returns a "RefPointData" object.
  The `kwargs` arguments are additional keyword arguments for `RefPointData`.
  These are the same as the `RefPointData_kwargs` used in [`PointCloudSolver`](@ref).
"""
function PointCloudBasis(element_type, polydeg; approximation_type = RBF(),
                         kwargs...)
    RefPointData(element_type, approximation_type, polydeg; kwargs...)
end

########################################
#            PointCloudDomain
########################################

# now that `PointCloudSolver` is defined, we can define constructors for `PointCloudDomain` which use `solver::PointCloudSolver`

function PointCloudDomain(solver::PointCloudSolver, points::Vector{Tv},
                          neighbors::Vector{Vector{Ti}},
                          boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {
                                                                                    N,
                                                                                    Tv <:
                                                                                    SVector{N,
                                                                                            Float64},
                                                                                    Ti
                                                                                    }
    return PointCloudDomain{NDIMS, typeof(points), typeof(neighbors),
                            typeof(boundary_tags)}(points, neighbors, boundary_tags)
end

function PointCloudDomain(solver::PointCloudSolver, pd::PointData{NDIMS},
                          boundary_tags::Dict{Symbol, BoundaryData{Ti, Tv}}) where {
                                                                                    NDIMS,
                                                                                    Tv <:
                                                                                    SVector{NDIMS,
                                                                                            Float64},
                                                                                    Ti
                                                                                    }
    return PointCloudDomain(pd, boundary_tags)
end

# Main function for creating PointCloudDomain
# We directly read in instead of generating
"""
    PointCloudDomain(solver::PointCloudSolver, filename::String)

- `solver::PointCloudSolver` contains information associated with the reference element (e.g., quadrature,
  basis evaluation, differentiation, etc).
- `filename` is a path specifying a `.mesh` file generated by
  [Medusa](https://gitlab.com/e62Lab/medusa).
"""
### Need to either pass basis from solve to read_medusa_file or calculate the 
# neighbors here. Probably makes more sense to initialize pd with medusa data
# then calculate neighbors and add here since that is how boundary tags are 
# treated as well. Big question is how to translate from expected "MeshData"
# md object initialization and use to pd instead. 
function PointCloudDomain(solver::PointCloudSolver{NDIMS},
                          filename::String,
                          boundary_names_dict::Dict{Symbol, Int}) where {NDIMS}
    # medusa_data, interior_idx, boundary_idxs, boundary_normals = read_medusa_file(filename)
    # pd = PointData(medusa_data, solver.basis)
    # boundary_tags = Dict(name => BoundaryData(boundary_idxs[idx], boundary_normals[idx])
    #                      for (name, idx) in boundary_names_dict)
    # return PointCloudDomain(solver, pd,
    #                         boundary_tags)

    return PointCloudDomain(solver.basis, filename,
                            boundary_names_dict)
end

# No checks for these meshes yet available
function Trixi.check_periodicity_mesh_boundary_conditions(mesh::PointCloudDomain,
                                                          boundary_conditions)
end

# Specialize printing for MPI and CUDA versions of PointCloudDomain
function Base.show(io::IO, solver::PointCloudSolver)
    @nospecialize solver # reduce precompilation time

    print(io, "PointCloudSolver{", real(solver), "}(")
    print(io, solver.basis)
    print(io, ", ", solver.engine)
    print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", solver::PointCloudSolver)
    @nospecialize solver # reduce precompilation time

    if get(io, :compact, false)
        show(io, solver)
    else
        summary_header(io, "PointCloudSolver{" * string(real(solver)) * "}")
        summary_line(io, "basis", solver.basis)
        summary_line(io, "engine",
                     solver.engine |> typeof |> nameof)
        if !(solver.engine isa AbstractRBFEngine)
            show(increment_indent(io), mime, solver.engine)
        end
        summary_footer(io)
    end
end

Base.summary(io::IO, solver::PointCloudSolver) = print(io,
                                                       "PointCloudSolver(" *
                                                       summary(solver.basis) *
                                                       ")")
end # @muladd
