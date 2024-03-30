# Based on Trixi/src/solvers/dg.jl
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
abstract type AbstractRBFEngine end

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

- Kopriva (2009)
  Implementing Spectral Methods for Partial Differential Equations:
  Algorithms for Scientists and Engineers
  [doi: 10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
- Hesthaven, Warburton (2007)
  Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and
  Applications
  [doi: 10.1007/978-0-387-72067-8](https://doi.org/10.1007/978-0-387-72067-8)

`RBFFDEngine()` is only implemented for conserved terms as
non-conservative terms should always be discretized in conjunction with a flux-splitting scheme,
see [`VolumeIntegralFluxDifferencing`](@ref).
This treatment is required to achieve, e.g., entropy-stability or well-balancedness.
"""
struct RBFFDEngine <: AbstractRBFEngine end

# # Example "engine" that utilizes indicators
# """
#     VolumeIntegralShockCapturingHG(indicator; volume_flux_dg=flux_central,
#                                               volume_flux_fv=flux_lax_friedrichs)

# Shock-capturing volume integral type for DG methods using a convex blending of
# the finite volume method with numerical flux `volume_flux_fv` and the
# [`VolumeIntegralFluxDifferencing`](@ref) with volume flux `volume_flux_dg`.
# The amount of blending is determined by the `indicator`, e.g.,
# [`IndicatorHennemannGassner`](@ref).

# ## References

# - Hennemann, Gassner (2020)
#   "A provably entropy stable subcell shock capturing approach for high order split form DG"
#   [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
# """
# struct VolumeIntegralShockCapturingHG{VolumeFluxDG, VolumeFluxFV, Indicator} <:
#        AbstractVolumeIntegral
#     volume_flux_dg::VolumeFluxDG # symmetric, e.g. split-form or entropy-conservative
#     volume_flux_fv::VolumeFluxFV # non-symmetric in general, e.g. entropy-dissipative
#     indicator::Indicator
# end

create_cache(mesh, equations, ::RBFFDEngine, solver, uEltype) = NamedTuple()
"""
    RBFSolver(; basis, mortar, surface_integral, engine)

Create a discontinuous Galerkin method.
If [`basis isa LobattoLegendreBasis`](@ref LobattoLegendreBasis),
this creates a [`RBFSolverSEM`](@ref).
"""
struct RBFSolver{Basis, RBFEngine}
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

function Base.show(io::IO, mime::MIME"text/plain", solver::RBFSolver)
    @nospecialize solver # reduce precompilation time

    if get(io, :compact, false)
        show(io, solver)
    else
        summary_header(io, "RBFSolver{" * string(real(solver)) * "}")
        summary_line(io, "basis", solver.basis)
        summary_line(io, "engine",
                     solver.engine |> typeof |> nameof)
        if !(solver.engine isa RBFFDEngine)
            show(increment_indent(io), mime, solver.engine)
        end
        summary_footer(io)
    end
end

Base.summary(io::IO, solver::RBFSolver) = print(io,
                                                "RBFSolver(" * summary(solver.basis) *
                                                ")")

@inline Base.real(solver::RBFSolver) = real(solver.basis)

function get_element_variables!(element_variables, u, domain, equations,
                                solver::RBFSolver, cache)
    get_element_variables!(element_variables, u, domain, equations,
                           solver.engine,
                           solver, cache)
end

function get_node_variables!(node_variables, domain, equations, solver::RBFSolver,
                             cache)
    get_node_variables!(node_variables, domain, equations, solver.engine,
                        solver,
                        cache)
end

# const MeshesRBFSolverSEM = Union{TreeMesh, StructuredMesh, UnstructuredMesh2D,
#                                  P4estMesh,
#                                  T8codeMesh}

# @inline function ndofs(domain::MeshesRBFSolverSEM, solver::RBFSolver, cache)
#     nelements(cache.elements) * nnodes(solver)^ndims(domain)
# end

# TODO: Taal performance, 1:nnodes(solver) vs. Base.OneTo(nnodes(solver)) vs. SOneTo(nnodes(solver)) for RBFSolverSEM
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

"""
    eachmortar(solver::RBFSolver, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the mortars in `cache`.
In particular, not the mortars themselves are returned.
"""
@inline eachmortar(solver::RBFSolver, cache) = Base.OneTo(nmortars(solver, cache))

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
@inline nmortars(solver::RBFSolver, cache) = nmortars(cache.mortars)
@inline nmpiinterfaces(solver::RBFSolver, cache) = nmpiinterfaces(cache.mpi_interfaces)
@inline nmpimortars(solver::RBFSolver, cache) = nmpimortars(cache.mpi_mortars)

# The following functions assume an array-of-structs memory layout
# We would like to experiment with different memory layout choices
# in the future, see
# - https://github.com/trixi-framework/Trixi.jl/issues/88
# - https://github.com/trixi-framework/Trixi.jl/issues/87
# - https://github.com/trixi-framework/Trixi.jl/issues/86
@inline function get_node_coords(x, equations, solver::RBFSolver, indices...)
    SVector(ntuple(@inline(idx->x[idx, indices...]), Val(ndims(equations))))
end

@inline function get_node_vars(u, equations, solver::RBFSolver, indices...)
    # There is a cut-off at `n == 10` inside of the method
    # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
    # in Julia `v1.5`, leading to type instabilities if
    # more than ten variables are used. That's why we use
    # `Val(...)` below.
    # We use `@inline` to make sure that the `getindex` calls are
    # really inlined, which might be the default choice of the Julia
    # compiler for standard `Array`s but not necessarily for more
    # advanced array types such as `PtrArray`s, cf.
    # https://github.com/JuliaSIMD/VectorizationBase.jl/issues/55
    SVector(ntuple(@inline(v->u[v, indices...]), Val(nvariables(equations))))
end

@inline function get_surface_node_vars(u, equations, solver::RBFSolver, indices...)
    # There is a cut-off at `n == 10` inside of the method
    # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
    # in Julia `v1.5`, leading to type instabilities if
    # more than ten variables are used. That's why we use
    # `Val(...)` below.
    u_ll = SVector(ntuple(@inline(v->u[1, v, indices...]), Val(nvariables(equations))))
    u_rr = SVector(ntuple(@inline(v->u[2, v, indices...]), Val(nvariables(equations))))
    return u_ll, u_rr
end

@inline function set_node_vars!(u, u_node, equations, solver::RBFSolver, indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u_node[v]
    end
    return nothing
end

@inline function add_to_node_vars!(u, u_node, equations, solver::RBFSolver, indices...)
    for v in eachvariable(equations)
        u[v, indices...] += u_node[v]
    end
    return nothing
end

# Use this function instead of `add_to_node_vars` to speed up
# multiply-and-add-to-node-vars operations
# See https://github.com/trixi-framework/Trixi.jl/pull/643
@inline function multiply_add_to_node_vars!(u, factor, u_node, equations,
                                            solver::RBFSolver,
                                            indices...)
    for v in eachvariable(equations)
        u[v, indices...] = u[v, indices...] + factor * u_node[v]
    end
    return nothing
end

# Used for analyze_solution
SolutionAnalyzer(solver::RBFSolver; kwargs...) = SolutionAnalyzer(solver.basis;
                                                                  kwargs...)

AdaptorAMR(domain, solver::RBFSolver) = AdaptorL2(solver.basis)

# General structs for discretizations based on the basic principle of
# RBFSolverSEM (discontinuous Galerkin spectral element method)
# include("solversem/solversem.jl")

# Finite difference methods using summation by parts (SBP) operators
# These methods are very similar to RBFSolver methods since they also impose interface
# and boundary conditions weakly. Thus, these methods can re-use a lot of
# functionality implemented for RBFSolverSEM.
# include("fdsbp_tree/fdsbp.jl")
# include("fdsbp_unstructured/fdsbp.jl")

function allocate_coefficients(domain::AbstractDomain, equations, solver::RBFSolver,
                               cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(cache.elements),
          nvariables(equations) * nnodes(solver)^ndims(domain) *
          nelements(solver, cache))
end

# @inline function wrap_array(u_ode::AbstractVector, domain::AbstractDomain, equations,
#                             solver::RBFSolverSEM, cache)
#     @boundscheck begin
#         @assert length(u_ode) ==
#                 nvariables(equations) * nnodes(solver)^ndims(domain) *
#                 nelements(solver, cache)
#     end
#     # We would like to use
#     #     reshape(u_ode, (nvariables(equations), ntuple(_ -> nnodes(solver), ndims(domain))..., nelements(solver, cache)))
#     # but that results in
#     #     ERROR: LoadError: cannot resize array with shared data
#     # when we resize! `u_ode` during AMR.
#     #
#     # !!! danger "Segfaults"
#     #     Remember to `GC.@preserve` temporaries such as copies of `u_ode`
#     #     and other stuff that is only used indirectly via `wrap_array` afterwards!

#     # Currently, there are problems when AD is used with `PtrArray`s in broadcasts
#     # since LoopVectorization does not support `ForwardDiff.Dual`s. Hence, we use
#     # optimized `PtrArray`s whenever possible and fall back to plain `Array`s
#     # otherwise.
#     if LoopVectorization.check_args(u_ode)
#         # This version using `PtrArray`s from StrideArrays.jl is very fast and
#         # does not result in allocations.
#         #
#         # !!! danger "Heisenbug"
#         #     Do not use this code when `@threaded` uses `Threads.@threads`. There is
#         #     a very strange Heisenbug that makes some parts very slow *sometimes*.
#         #     In fact, everything can be fast and fine for many cases but some parts
#         #     of the RHS evaluation can take *exactly* (!) five seconds randomly...
#         #     Hence, this version should only be used when `@threaded` is based on
#         #     `@batch` from Polyester.jl or something similar. Using Polyester.jl
#         #     is probably the best option since everything will be handed over to
#         #     Chris Elrod, one of the best performance software engineers for Julia.
#         PtrArray(pointer(u_ode),
#                  (StaticInt(nvariables(equations)),
#                   ntuple(_ -> StaticInt(nnodes(solver)), ndims(domain))...,
#                   nelements(solver, cache)))
#         #  (nvariables(equations), ntuple(_ -> nnodes(solver), ndims(domain))..., nelements(solver, cache)))
#     else
#         # The following version is reasonably fast and allows us to `resize!(u_ode, ...)`.
#         unsafe_wrap(Array{eltype(u_ode), ndims(domain) + 2}, pointer(u_ode),
#                     (nvariables(equations),
#                      ntuple(_ -> nnodes(solver), ndims(domain))...,
#                      nelements(solver, cache)))
#     end
# end

# # Finite difference summation by parts (FDSBP) methods
# @inline function wrap_array(u_ode::AbstractVector, domain::AbstractDomain, equations,
#                             solver::FDSBP, cache)
#     @boundscheck begin
#         @assert length(u_ode) ==
#                 nvariables(equations) * nnodes(solver)^ndims(domain) *
#                 nelements(solver, cache)
#     end
#     # See comments on the RBFSolverSEM version above
#     if LoopVectorization.check_args(u_ode)
#         # Here, we do not specialize on the number of nodes using `StaticInt` since
#         # - it will not be type stable (SBP operators just store it as a runtime value)
#         # - FD methods tend to use high node counts
#         PtrArray(pointer(u_ode),
#                  (StaticInt(nvariables(equations)),
#                   ntuple(_ -> nnodes(solver), ndims(domain))...,
#                   nelements(solver, cache)))
#     else
#         # The following version is reasonably fast and allows us to `resize!(u_ode, ...)`.
#         unsafe_wrap(Array{eltype(u_ode), ndims(domain) + 2}, pointer(u_ode),
#                     (nvariables(equations),
#                      ntuple(_ -> nnodes(solver), ndims(domain))...,
#                      nelements(solver, cache)))
#     end
# end

# General fallback
@inline function wrap_array(u_ode::AbstractVector, domain::AbstractDomain, equations,
                            solver::RBFSolver, cache)
    wrap_array_native(u_ode, domain, equations, solver, cache)
end

# Like `wrap_array`, but guarantees to return a plain `Array`, which can be better
# for interfacing with external C libraries (MPI, HDF5, visualization),
# writing solution files etc.
@inline function wrap_array_native(u_ode::AbstractVector, domain::AbstractDomain,
                                   equations,
                                   solver::RBFSolver, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                nvariables(equations) * nnodes(solver)^ndims(domain) *
                nelements(solver, cache)
    end
    unsafe_wrap(Array{eltype(u_ode), ndims(domain) + 2}, pointer(u_ode),
                (nvariables(equations), ntuple(_ -> nnodes(solver), ndims(domain))...,
                 nelements(solver, cache)))
end

function compute_coefficients!(u, func, t, domain::AbstractDomain{1}, equations,
                               solver::RBFSolver,
                               cache)
    @threaded for element in eachelement(solver, cache)
        for i in eachnode(solver)
            x_node = get_node_coords(cache.elements.node_coordinates, equations, solver,
                                     i,
                                     element)
            # Changing the node positions passed to the initial condition by the minimum
            # amount possible with the current type of floating point numbers allows setting
            # discontinuous initial data in a simple way. In particular, a check like `if x < x_jump`
            # works if the jump location `x_jump` is at the position of an interface.
            if i == 1
                x_node = SVector(nextfloat(x_node[1]))
            elseif i == nnodes(solver)
                x_node = SVector(prevfloat(x_node[1]))
            end
            u_node = func(x_node, t, equations)
            set_node_vars!(u, u_node, equations, solver, i, element)
        end
    end
end

function compute_coefficients!(u, func, t, domain::AbstractDomain{2}, equations,
                               solver::RBFSolver,
                               cache)
    @threaded for element in eachelement(solver, cache)
        for j in eachnode(solver), i in eachnode(solver)
            x_node = get_node_coords(cache.elements.node_coordinates, equations, solver,
                                     i,
                                     j, element)
            u_node = func(x_node, t, equations)
            set_node_vars!(u, u_node, equations, solver, i, j, element)
        end
    end
end

function compute_coefficients!(u, func, t, domain::AbstractDomain{3}, equations,
                               solver::RBFSolver,
                               cache)
    @threaded for element in eachelement(solver, cache)
        for k in eachnode(solver), j in eachnode(solver), i in eachnode(solver)
            x_node = get_node_coords(cache.elements.node_coordinates, equations, solver,
                                     i,
                                     j, k, element)
            u_node = func(x_node, t, equations)
            set_node_vars!(u, u_node, equations, solver, i, j, k, element)
        end
    end
end

# Discretizations specific to each domain type of Trixi.jl
# If some functionality is shared by multiple combinations of meshes/solvers,
# it is defined in the directory of the most basic domain and solver type.
# The most basic solver type in Trixi.jl is RBFSolverSEM (historic reasons and background
# of the main contributors).
# We consider the `TreeMesh` to be the most basic domain type since it is Cartesian
# and was the first domain in Trixi.jl. The order of the other domain types is the same
# as the include order below.
# include("solversem_tree/solver.jl")
# include("solversem_structured/solver.jl")
# include("solversem_unstructured/solver.jl")
# include("solversem_p4est/solver.jl")
# include("solversem_t8code/solver.jl")
end # @muladd
