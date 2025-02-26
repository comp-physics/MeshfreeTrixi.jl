# Based on Trixi/src/solvers/dg.jl
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

abstract type AbstractRBFEngine end

abstract type ExecutionSpace end
struct CPUExecutionSpace <: ExecutionSpace end
struct CUDAExecutionSpace <: ExecutionSpace end

function get_element_variables!(element_variables, u, mesh, equations,
                                engine::AbstractRBFEngine, solver, cache)
    nothing
end

function get_node_variables!(node_variables, mesh, equations,
                             engine::AbstractRBFEngine, solver, cache)
    nothing
end
"""
    RBFFDEngine()

The classical RBF-FD backend type for meshfree methods as explained in
standard textbooks. Replaces VolumeIntegralWeakForm()

## References

- Flyer, Natasha (2016)
  Enhancing Finite Difference Methods with Radial Basis Functions:
  Experiments on the Navier-Stokes Equations
  [doi: 10.1016/j.jcp.2016.02.078](https://doi.org/10.1016/j.jcp.2016.02.078)

"""
# New "Engines" refer to different ways to perform Derivative calculation
# for example, we would define a new engine for Schafer sparse backend 
struct RBFFDEngine <: AbstractRBFEngine end

create_cache(mesh, equations, ::RBFFDEngine, solver, uEltype) = NamedTuple()

### MOVE TO DEPRECATE RBF_SOLVER. ADDS REDUNDANCY TO POINTCLOUDSOLVER WHICH ALREADY 
### CAN DISPATCH TO DIFFERENT BACKENDS BASED ON ENGINE
"""
    RBFSolver(; basis, engine)

Create an RBF-FD method.
"""
struct RBFSolver{Basis, RBFEngine, Space <: ExecutionSpace}
    basis::Basis
    engine::RBFEngine
end

function Base.show(io::IO, solver::RBFSolver)
    @nospecialize solver # reduce precompilation time

    print(io, "RBFSolver{", real(solver), "}(")
    print(io, solver.basis)
    print(io, ", ", solver.engine)
    print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain",
                   solver::RBFSolver{Basis, RBFEngine, Space}) where {Basis,
                                                                      RBFEngine,
                                                                      Space}
    @nospecialize solver # reduce precompilation time

    if get(io, :compact, false)
        show(io, solver)
    else
        summary_header(io, "RBFSolver{" * string(real(solver)) * "}")
        summary_line(io, "basis", solver.basis)
        summary_line(io, "engine",
                     solver.engine |> typeof |> nameof)
        if !(solver.engine isa AbstractRBFEngine)
            show(increment_indent(io), mime, solver.engine)
        end
        summary_line(io, "execution space",
                     Space)
        if !(solver.engine isa ExecutionSpace)
            show(increment_indent(io), mime, Space)
        end
        summary_footer(io)
    end
end

Base.summary(io::IO, solver::RBFSolver) = print(io,
                                                "RBFSolver(" * summary(solver.basis) *
                                                ")")

@inline Base.real(solver::RBFSolver) = real(solver.basis)

# const MeshesRBFSolverSEM = Union{TreeMesh, StructuredMesh, UnstructuredMesh2D,
#                                  P4estMesh,
#                                  T8codeMesh}

# @inline function ndofs(domain::MeshesRBFSolverSEM, solver::RBFSolver, cache)
#     nelements(cache.elements) * nnodes(solver)^ndims(domain)
# end

# TODO: Generic methods below need cleanup. Most are unused or specialized in PointCloudSolver 
# We would only want generic methods for fallback between point based solvers and 
# cell based solvers.
"""
    eachnode(solver::RBFSolver)

Return an iterator over the indices that specify the location in relevant data structures
for the nodes in `solver`.
In particular, not the nodes themselves are returned.
"""
@inline eachnode(solver::RBFSolver) = Base.OneTo(nnodes(solver))
@inline nnodes(solver::RBFSolver) = nnodes(solver.basis)

# This is used in some more general analysis code and needs to dispatch on the
# `domain` for some combinations of domain/solver.
@inline nelements(domain, solver::RBFSolver, cache) = nelements(solver, cache)
@inline function ndofsglobal(domain, solver::RBFSolver, cache)
    nelementsglobal(solver, cache) * nnodes(solver)^ndims(domain)
end

"""
    eachelement(solver::RBFSolver, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `cache`.
In particular, not the elements themselves are returned.
"""
@inline eachelement(solver::RBFSolver, cache) = Base.OneTo(nelements(solver, cache))

"""
    eachinterface(solver::RBFSolver, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the interfaces in `cache`.
In particular, not the interfaces themselves are returned.
"""
@inline eachinterface(solver::RBFSolver, cache) = Base.OneTo(ninterfaces(solver, cache))

"""
    eachboundary(solver::RBFSolver, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the boundaries in `cache`.
In particular, not the boundaries themselves are returned.
"""
@inline eachboundary(solver::RBFSolver, cache) = Base.OneTo(nboundaries(solver, cache))

# """
#     eachmortar(solver::RBFSolver, cache)

# Return an iterator over the indices that specify the location in relevant data structures
# for the mortars in `cache`.
# In particular, not the mortars themselves are returned.
# """
# @inline eachmortar(solver::RBFSolver, cache) = Base.OneTo(nmortars(solver, cache))

"""
    eachmpiinterface(solver::RBFSolver, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the MPI interfaces in `cache`.
In particular, not the interfaces themselves are returned.
"""
@inline eachmpiinterface(solver::RBFSolver, cache) = Base.OneTo(nmpiinterfaces(solver,
                                                                               cache))

"""
    eachmpimortar(solver::RBFSolver, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the MPI mortars in `cache`.
In particular, not the mortars themselves are returned.
"""
@inline eachmpimortar(solver::RBFSolver, cache) = Base.OneTo(nmpimortars(solver, cache))

@inline nelements(solver::RBFSolver, cache) = nelements(cache.elements)
@inline function nelementsglobal(solver::RBFSolver, cache)
    mpi_isparallel() ? cache.mpi_cache.n_elements_global : nelements(solver, cache)
end
@inline ninterfaces(solver::RBFSolver, cache) = ninterfaces(cache.interfaces)
@inline nboundaries(solver::RBFSolver, cache) = nboundaries(cache.boundaries)
# @inline nmortars(solver::RBFSolver, cache) = nmortars(cache.mortars)
@inline nmpiinterfaces(solver::RBFSolver, cache) = nmpiinterfaces(cache.mpi_interfaces)
# @inline nmpimortars(solver::RBFSolver, cache) = nmpimortars(cache.mpi_mortars)

# # The following functions assume an array-of-structs memory layout
# # We would like to experiment with different memory layout choices
# # in the future, see
# # - https://github.com/trixi-framework/Trixi.jl/issues/88
# # - https://github.com/trixi-framework/Trixi.jl/issues/87
# # - https://github.com/trixi-framework/Trixi.jl/issues/86
# @inline function get_node_coords(x, equations, solver::RBFSolver, indices...)
#     SVector(ntuple(@inline(idx->x[idx, indices...]), Val(ndims(equations))))
# end

# @inline function get_node_vars(u, equations, solver::RBFSolver, indices...)
#     # There is a cut-off at `n == 10` inside of the method
#     # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
#     # in Julia `v1.5`, leading to type instabilities if
#     # more than ten variables are used. That's why we use
#     # `Val(...)` below.
#     # We use `@inline` to make sure that the `getindex` calls are
#     # really inlined, which might be the default choice of the Julia
#     # compiler for standard `Array`s but not necessarily for more
#     # advanced array types such as `PtrArray`s, cf.
#     # https://github.com/JuliaSIMD/VectorizationBase.jl/issues/55
#     SVector(ntuple(@inline(v->u[v, indices...]), Val(nvariables(equations))))
# end

# @inline function get_surface_node_vars(u, equations, solver::RBFSolver, indices...)
#     # There is a cut-off at `n == 10` inside of the method
#     # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
#     # in Julia `v1.5`, leading to type instabilities if
#     # more than ten variables are used. That's why we use
#     # `Val(...)` below.
#     u_ll = SVector(ntuple(@inline(v->u[1, v, indices...]), Val(nvariables(equations))))
#     u_rr = SVector(ntuple(@inline(v->u[2, v, indices...]), Val(nvariables(equations))))
#     return u_ll, u_rr
# end

# @inline function set_node_vars!(u, u_node, equations, solver::RBFSolver, indices...)
#     for v in eachvariable(equations)
#         u[v, indices...] = u_node[v]
#     end
#     return nothing
# end

# @inline function add_to_node_vars!(u, u_node, equations, solver::RBFSolver, indices...)
#     for v in eachvariable(equations)
#         u[v, indices...] += u_node[v]
#     end
#     return nothing
# end

# # Use this function instead of `add_to_node_vars` to speed up
# # multiply-and-add-to-node-vars operations
# # See https://github.com/trixi-framework/Trixi.jl/pull/643
# @inline function multiply_add_to_node_vars!(u, factor, u_node, equations,
#                                             solver::RBFSolver,
#                                             indices...)
#     for v in eachvariable(equations)
#         u[v, indices...] = u[v, indices...] + factor * u_node[v]
#     end
#     return nothing
# end

# Used for analyze_solution
SolutionAnalyzer(solver::RBFSolver; kwargs...) = SolutionAnalyzer(solver.basis;
                                                                  kwargs...)

AdaptorAMR(domain, solver::RBFSolver) = AdaptorL2(solver.basis)
end # @muladd
