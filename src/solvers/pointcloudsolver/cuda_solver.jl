# Based on Trixi/solvers/DGMulti/types.jl
# Rewritten to work with RBF-FD methods on PointCloudDomains.
# Note: we define type aliases outside of the @muladd block to avoid Revise breaking when code
# inside the @muladd block is edited. See https://github.com/trixi-framework/Trixi.jl/issues/801
# for more details.

# `PointCloudSolver` refers to both multiple RBFSolver types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
# CUDAPointCloudSolver allows specialization for Nvidia GPUs
struct CUDAPointCloudSolver{NDIMS, ElemType, ApproxType, Engine} <:
       PointCloudSolver{NDIMS, ElemType, ApproxType, Engine}
end

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# ALL METHODS FOR MUL NEED TO BE CONVERTED TO CUDAPointCloudSolver
# out .<- A .* x
mul_by!(A::AbstractVector) = @inline (out, x) -> out .= A .* x

# out <- A*x
mul_by!(A) = @inline (out, x) -> mul!(out, A, x)
mul_by!(A::AbstractSparseMatrix) = @inline (out, x) -> mul!(out, A, x)
function mul_by!(A::LinearAlgebra.AdjOrTrans{T, S}) where {T, S <: AbstractSparseMatrix}
    @inline (out, x) -> mul!(out, A, x)
end

#  out <- out + α * A * x
mul_by_accum!(A, α) = @inline (out, x) -> mul!(out, A, x, α, One())
function mul_by_accum!(A::AbstractSparseMatrix, α)
    @inline (out, x) -> mul!(out, A, x, α, One())
end

# out <- out + A * x
mul_by_accum!(A) = mul_by_accum!(A, One())

# StructArray fallback
### How to implement fallback for CuSparseMatrixCSC * CuArray
### since we are removing StructArrays
@inline function apply_to_each_field(f::F, args::Vararg{Any, N}) where {F, N}
    # StructArrays.foreachfield(f, args...) ### Convert to CUDAPointCloudSolver
    # We just apply to all columns so we just call f?
    f(args...)
end

# Convenience Methods for reseting CuArray
function set_to_zero!(array::CuArray)
    array .= zero(eltype(array)) # Convert to CUDAPointCloudSolver
end

# # iteration over quantities over the entire domain (dofs, quad nodes, face nodes).
# """
#     each_dof_global(domain::PointCloudDomain, solver::PointCloudSolver, other_args...)

# Return an iterator over the indices that specify the location in relevant data structures
# for the degrees of freedom (DOF) in `solver`.
# In particular, not the DOFs themselves are returned.
# """
# @inline function each_dof_global(domain::PointCloudDomain, solver::PointCloudSolver,
#                                  other_args...)
#     Base.OneTo(ndofs(domain, solver, other_args...))
# end

# interface with semidiscretization_hyperbolic
# function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
#                                     domain::PointCloudDomain,
#                                     solver::PointCloudSolver,
#                                     cache) where {Keys, ValueTypes <: NTuple{N, Any}
#                                                   } where {N}
#     return boundary_conditions
# end ### fallback to methods for pointcloudsolver

# Allocate nested array type for CUDAPointCloudSolver solution storage.
function allocate_nested_array(uEltype, nvars, array_dimensions, solver)
    # store components as columns in matrix
    return CuArray(zeros(uEltype, array_dimensions..., nvars))
end

function reset_du!(du, solver::CUDAPointCloudSolver, other_args...)
    # @threaded for i in eachindex(du)
    #     du[i] = zero(eltype(du))
    # end
    du .= zero(eltype(du))

    return du
end

# Constructs cache variables for PointCloudDomains
# specialized for CUDAPointCloudSolver
function Trixi.create_cache(domain::PointCloudDomain{NDIMS}, equations,
                            solver::CUDAPointCloudSolver, RealT,
                            uEltype) where {NDIMS}
    rd = solver.basis
    pd = domain.pd

    # CHANGE TO CUDAPointCloudSolver compat

    # differentiation matrices as required by the governing equation
    # This will compute diff_mat with two entries, the first being the dx,
    # and the second being dy. 
    rbf_differentiation_matrices = CuSparseMatrixCSC.(compute_flux_operator(solver,
                                                                            domain))

    nvars = nvariables(equations)

    # storage for volume quadrature values, face quadrature values, flux values
    # currently keeping all of these but may not need all of them
    u_values = allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)
    u_face_values = allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)
    flux_face_values = allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)

    # local storage for volume integral and source computations
    local_values_threaded = [allocate_nested_array(uEltype, nvars, (pd.num_points,),
                                                   solver)
                             for _ in 1:Threads.nthreads()]

    # for scaling by curved geometric terms (not used by affine PointCloudDomain)
    # We use these as scratch space instead
    flux_threaded = [[allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)
                      for _ in 1:NDIMS] for _ in 1:Threads.nthreads()]
    rhs_local_threaded = [allocate_nested_array(uEltype, nvars,
                                                (pd.num_points,), solver)
                          for _ in 1:Threads.nthreads()]

    return (; pd, rbf_differentiation_matrices,
            u_values, u_face_values, flux_face_values,
            local_values_threaded, flux_threaded, rhs_local_threaded)
end

# function Trixi.allocate_coefficients(domain::PointCloudDomain, equations,
#                                      solver::PointCloudSolver, cache)
#     return allocate_nested_array(real(solver), nvariables(equations),
#                                  (domain.pd.num_points,),
#                                  solver)
# end # fallback to pointcloudsolver

function Trixi.compute_coefficients!(u, initial_condition, t,
                                     domain::PointCloudDomain, equations,
                                     solver::CUDAPointCloudSolver, cache)
    pd = domain.pd
    rd = solver.basis
    @unpack u_values = cache

    # CHANGE TO CUDAPointCloudSolver compat
    ### MAY NEED TO CONVERT TO KERNEL

    for i in eachelement(domain, solver, cache)
        u[i, :] = initial_condition(pd.points[i],
                                    t, equations)
    end
end

function calc_fluxes!(du, u, domain::PointCloudDomain,
                      have_nonconservative_terms::False, equations,
                      engine::RBFFDEngine,
                      solver::CUDAPointCloudSolver,
                      cache)
    rd = solver.basis
    pd = domain.pd
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded = cache

    # CHANGE TO CUDAPointCloudSolver compat

    flux_values = local_values_threaded[1]
    for i in eachdim(domain)
        for e in eachelement(domain, solver, cache)
            flux_values[e] = flux(u[e], i, equations)
        end
        apply_to_each_field(mul_by_accum!(rbf_differentiation_matrices[i],
                                          -1),
                            du, flux_values)
    end
end

# do nothing for periodic (default) boundary conditions
# change signature to include du, u in order 
# to implement Dirichlet BCs
# function calc_boundary_flux!(du, u, cache, t,
#                              boundary_conditions::BoundaryConditionPeriodic,
#                              domain, have_nonconservative_terms, equations,
#                              solver::CUDAPointCloudSolver)
#     nothing
# end

function calc_boundary_flux!(du, u, cache, t, boundary_conditions, domain,
                             have_nonconservative_terms, equations,
                             solver::CUDAPointCloudSolver)
    for (key, value) in zip(keys(boundary_conditions), boundary_conditions)
        calc_single_boundary_flux!(du, u, cache, t, value,
                                   key,
                                   domain, have_nonconservative_terms, equations,
                                   solver)
    end
end # likely can fallback to pointcloudsolver

function calc_single_boundary_flux!(du, u, cache, t, boundary_condition, boundary_key,
                                    domain,
                                    have_nonconservative_terms::False, equations,
                                    solver::CUDAPointCloudSolver{NDIMS}) where {NDIMS}
    rd = solver.basis
    pd = domain.pd
    @unpack u_face_values, flux_face_values, local_values_threaded = cache

    # CHANGE TO CUDAPointCloudSolver compat
    ### MAY NEED TO CONVERT TO KERNEL

    boundary_flux = local_values_threaded[1]
    set_to_zero!(boundary_flux)

    # Extract boundary elements
    boundary_idxs = domain.boundary_tags[boundary_key].idx
    boundary_normals = domain.boundary_tags[boundary_key].normals

    # Modified to strongly impose BCs
    # Requires mutating u and setting du
    # to 0 at boundary locations
    for i in eachindex(boundary_idxs)
        boundary_idx = boundary_idxs[i]
        boundary_normal = boundary_normals[i]
        boundary_coordinates = pd.points[boundary_idx]
        u_boundary = u[boundary_idx]
        du[boundary_idx], u[boundary_idx] = boundary_condition(du[boundary_idx],
                                                               u[boundary_idx],
                                                               boundary_normal,
                                                               boundary_coordinates,
                                                               t,
                                                               FluxZero(), equations)
    end
end

# Multiple calc_sources! to resolve method ambiguities
function calc_sources!(du, u, t, source_terms::Nothing,
                       domain, equations, solver::CUDAPointCloudSolver, cache)
    nothing
end

# Redefined to allow for generic sources include sources 
# requiring operator application. Each source will
# be a callable struct containing its own caches
function calc_sources!(du, u, t, source_terms,
                       domain, equations, solver::CUDAPointCloudSolver, cache)

    # CHANGE TO CUDAPointCloudSolver compat
    ### NOTE: May actually require direct changes to each source term
    ### as this only loops through the different sources
    for source in values(source_terms)
        save_tag = string(nameof(typeof(source)))
        @trixi_timeit timer() "calc $save_tag" source(du, u, t, domain, equations,
                                                      solver, cache)
    end
end

function Trixi.rhs!(du, u, t, domain, equations,
                    initial_condition, boundary_conditions::BC, source_terms::Source,
                    solver::CUDAPointCloudSolver, cache) where {BC, Source}
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, solver, cache)

    # Require two passes for strongly imposed BCs
    # First sets u to BC value, 
    # second sets du to zero
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(du, u, cache, t, boundary_conditions, domain,
                            have_nonconservative_terms(equations), equations, solver)
    end

    @trixi_timeit timer() "calc fluxes" begin
        calc_fluxes!(du, u, domain,
                     have_nonconservative_terms(equations), equations,
                     solver.engine, solver, cache)
    end

    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, domain, equations, solver, cache)
    end

    # Require two passes for strongly imposed BCs
    # second sets du to zero
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(du, u, cache, t, boundary_conditions, domain,
                            have_nonconservative_terms(equations), equations, solver)
    end

    return nothing
end
end # @muladd
