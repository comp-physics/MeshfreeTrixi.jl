# Based on Trixi/src/solvers/DGMulti/dg.jl
# Rewritten to work with RBF-FD methods on PointCloudDomains.
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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

# specialize for SBP operators since `matmul!` doesn't work for `UniformScaling` types.
struct MulByUniformScaling end
struct MulByAccumUniformScaling end
mul_by!(A::UniformScaling) = MulByUniformScaling()
mul_by_accum!(A::UniformScaling) = MulByAccumUniformScaling()

# StructArray fallback
@inline function apply_to_each_field(f::F, args::Vararg{Any, N}) where {F, N}
    StructArrays.foreachfield(f, args...)
end

# specialize for UniformScaling types: works for either StructArray{SVector} or Matrix{SVector}
# solution storage formats.
@inline apply_to_each_field(f::MulByUniformScaling, out, x, args...) = copy!(out, x)
@inline function apply_to_each_field(f::MulByAccumUniformScaling, out, x, args...)
    @threaded for i in eachindex(x)
        out[i] = out[i] + x[i]
    end
end

# Convenience Methods for reseting struct array
function set_to_zero!(array::StructArray)
    StructArrays.foreachfield(col -> fill!(col, 0.0), array)
end

"""
    eachdim(domain)

Return an iterator over the indices that specify the location in relevant data structures
for the dimensions in `AbstractTree`.
In particular, not the dimensions themselves are returned.
"""
@inline eachdim(domain) = Base.OneTo(ndims(domain))

# iteration over all elements in a domain
@inline function Trixi.ndofs(domain::PointCloudDomain, solver::PointCloudSolver,
                             other_args...)
    domain.pd.num_points
end
"""
    eachelement(domain::PointCloudDomain, solver::PointCloudSolver, other_args...)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `domain`.
In particular, not the elements themselves are returned.
"""
@inline function eachelement(domain::PointCloudDomain, solver::PointCloudSolver,
                             other_args...)
    Base.OneTo(domain.pd.num_points)
end

# iteration over quantities in a single element
@inline nnodes(basis::RefPointData) = basis.nv # synonymous for number of neighbors

get_component(u::StructArray, i::Int) = StructArrays.component(u, i)
get_component(u::AbstractArray{<:SVector}, i::Int) = getindex.(u, i)

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
Trixi.wrap_array(u_ode, domain::PointCloudDomain, equations, solver::PointCloudSolver, cache) = u_ode
Trixi.wrap_array_native(u_ode, domain::PointCloudDomain, equations, solver::PointCloudSolver, cache) = u_ode
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
                                    domain::PointCloudDomain,
                                    solver::PointCloudSolver,
                                    cache) where {Keys, ValueTypes <: NTuple{N, Any}
                                                  } where {N}
    return boundary_conditions
end

# Allocate nested array type for PointCloudSolver solution storage.
function allocate_nested_array(uEltype, nvars, array_dimensions, solver)
    # store components as separate arrays, combine via StructArrays
    return StructArray{SVector{nvars, uEltype}}(ntuple(_ -> zeros(uEltype,
                                                                  array_dimensions...),
                                                       nvars))
end

function reset_du!(du, solver::PointCloudSolver, other_args...)
    @threaded for i in eachindex(du)
        du[i] = zero(eltype(du))
    end

    return du
end

# Constructs cache variables for PointCloudDomains
# Needs heavy rework to be compatible with PointCloudSolver
# Esp. since we need to generate operator matrices here
# that depend on our governing equation
function Trixi.create_cache(domain::PointCloudDomain{NDIMS}, equations,
                            solver::PointCloudSolver, RealT,
                            uEltype) where {NDIMS}
    rd = solver.basis
    pd = domain.pd

    # differentiation matrices as required by the governing equation
    # This will compute diff_mat with two entries, the first being the dx,
    # and the second being dy. 
    rbf_differentiation_matrices = compute_flux_operator(solver, domain)

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

function Trixi.allocate_coefficients(domain::PointCloudDomain, equations,
                                     solver::PointCloudSolver, cache)
    return allocate_nested_array(real(solver), nvariables(equations),
                                 (domain.pd.num_points,),
                                 solver)
end

function Trixi.compute_coefficients!(u, initial_condition, t,
                                     domain::PointCloudDomain, equations,
                                     solver::PointCloudSolver, cache)
    pd = domain.pd
    rd = solver.basis
    @unpack u_values = cache

    for i in eachelement(domain, solver, cache)
        u[i] = initial_condition(pd.points[i],
                                 t, equations)
    end
end

### All max stepsize callback functions need to be redefined 
### to work with PointCloudSolver
# estimates the timestep based on polynomial degree and domain. Does not account for physics (e.g.,
# computes an estimate of `dt` based on the advection equation with constant unit advection speed).
# function estimate_dt(domain::PointCloudDomain, solver::PointCloudSolver)
#     rd = solver.basis # RefPointData
#     return StartUpDG.estimate_h(rd, domain.pd) / StartUpDG.inverse_trace_constant(rd)
# end

# dt_polydeg_scaling(solver::PointCloudSolver) = inv(solver.basis.N + 1)
# function dt_polydeg_scaling(solver::PointCloudSolver{3, <:Wedge, <:TensorProductWedge})
#     inv(maximum(solver.basis.N) + 1)
# end

# # for the stepsize callback
# function max_dt(u, t, domain::PointCloudDomain,
#                 constant_speed::False, equations, solver::PointCloudSolver{NDIMS},
#                 cache) where {NDIMS}
#     @unpack pd = domain
#     rd = solver.basis

#     dt_min = Inf
#     for e in eachelement(domain, solver, cache)
#         h_e = StartUpDG.estimate_h(e, rd, pd)
#         max_speeds = ntuple(_ -> nextfloat(zero(t)), NDIMS)
#         for i in Base.OneTo(rd.Np) # loop over nodes
#             lambda_i = max_abs_speeds(u[i, e], equations)
#             max_speeds = max.(max_speeds, lambda_i)
#         end
#         dt_min = min(dt_min, h_e / sum(max_speeds))
#     end
#     # This mimics `max_dt` for `TreeMesh`, except that `nnodes(solver)` is replaced by
#     # `polydeg+1`. This is because `nnodes(solver)` returns the total number of
#     # multi-dimensional nodes for PointCloudSolver solver types, while `nnodes(solver)` returns
#     # the number of 1D nodes for `DGSEM` solvers.
#     return 2 * dt_min * dt_polydeg_scaling(solver)
# end

# function max_dt(u, t, domain::PointCloudDomain,
#                 constant_speed::True, equations, solver::PointCloudSolver{NDIMS},
#                 cache) where {NDIMS}
#     @unpack pd = domain
#     rd = solver.basis

#     dt_min = Inf
#     for e in eachelement(domain, solver, cache)
#         h_e = StartUpDG.estimate_h(e, rd, pd)
#         max_speeds = ntuple(_ -> nextfloat(zero(t)), NDIMS)
#         for i in Base.OneTo(rd.Np) # loop over nodes
#             max_speeds = max.(max_abs_speeds(equations), max_speeds)
#         end
#         dt_min = min(dt_min, h_e / sum(max_speeds))
#     end
#     # This mimics `max_dt` for `TreeMesh`, except that `nnodes(solver)` is replaced by
#     # `polydeg+1`. This is because `nnodes(solver)` returns the total number of
#     # multi-dimensional nodes for PointCloudSolver solver types, while `nnodes(solver)` returns
#     # the number of 1D nodes for `DGSEM` solvers.
#     return 2 * dt_min * dt_polydeg_scaling(solver)
# end

function calc_fluxes!(du, u, domain::PointCloudDomain,
                      have_nonconservative_terms::False, equations,
                      engine::RBFFDEngine,
                      solver::PointCloudSolver,
                      cache)
    rd = solver.basis
    pd = domain.pd
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded = cache

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
#                              solver::PointCloudSolver)
#     nothing
# end

function calc_boundary_flux!(du, u, cache, t, boundary_conditions, domain,
                             have_nonconservative_terms, equations,
                             solver::PointCloudSolver)
    for (key, value) in zip(keys(boundary_conditions), boundary_conditions)
        calc_single_boundary_flux!(du, u, cache, t, value,
                                   key,
                                   domain, have_nonconservative_terms, equations,
                                   solver)
    end
end

function calc_single_boundary_flux!(du, u, cache, t, boundary_condition, boundary_key,
                                    domain,
                                    have_nonconservative_terms::False, equations,
                                    solver::PointCloudSolver{NDIMS}) where {NDIMS}
    rd = solver.basis
    pd = domain.pd
    @unpack u_face_values, flux_face_values, local_values_threaded = cache

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

### Nonconservative BCs not implemented yet
### Would only require operating on primitive var instead
# function calc_single_boundary_flux!(du, u, cache, t, boundary_condition, boundary_key,
#                                     domain,
#                                     have_nonconservative_terms::True, equations,
#                                     solver::PointCloudSolver{NDIMS}) where {NDIMS}
#     rd = solver.basis
#     pd = domain.pd
#     surface_flux, nonconservative_flux = solver.surface_integral.surface_flux

#     # reshape face/normal arrays to have size = (num_points_on_face, num_faces_total).
#     # domain.boundary_faces indexes into the columns of these face-reshaped arrays.
#     num_pts_per_face = rd.Nfq ÷ StartUpDG.num_faces(rd.element_type)
#     num_faces_total = StartUpDG.num_faces(rd.element_type) * pd.num_elements

#     # This function was originally defined as
#     # `reshape_by_face(u) = reshape(view(u, :), num_pts_per_face, num_faces_total)`.
#     # This results in allocations due to https://github.com/JuliaLang/julia/issues/36313.
#     # To avoid allocations, we use Tim Holy's suggestion:
#     # https://github.com/JuliaLang/julia/issues/36313#issuecomment-782336300.
#     reshape_by_face(u) = Base.ReshapedArray(u, (num_pts_per_face, num_faces_total), ())

#     u_face_values = reshape_by_face(cache.u_face_values)
#     flux_face_values = reshape_by_face(cache.flux_face_values)
#     Jf = reshape_by_face(pd.Jf)
#     nxyzJ, xyzf = reshape_by_face.(pd.nxyzJ), reshape_by_face.(pd.xyzf) # broadcast over nxyzJ::NTuple{NDIMS,Matrix}

#     # loop through boundary faces, which correspond to columns of reshaped u_face_values, ...
#     for f in domain.boundary_faces[boundary_key]
#         for i in Base.OneTo(num_pts_per_face)
#             face_normal = SVector{NDIMS}(getindex.(nxyzJ, i, f)) / Jf[i, f]
#             face_coordinates = SVector{NDIMS}(getindex.(xyzf, i, f))

#             # Compute conservative and non-conservative fluxes separately.
#             # This imposes boundary conditions on the conservative part of the flux.
#             cons_flux_at_face_node = boundary_condition(u_face_values[i, f],
#                                                         face_normal, face_coordinates,
#                                                         t,
#                                                         surface_flux, equations)

#             # Compute pointwise nonconservative numerical flux at the boundary.
#             # In general, nonconservative fluxes can depend on both the contravariant
#             # vectors (normal direction) at the current node and the averaged ones.
#             # However, there is only one `face_normal` at boundaries, which we pass in twice.
#             # Note: This does not set any type of boundary condition for the nonconservative term
#             noncons_flux_at_face_node = nonconservative_flux(u_face_values[i, f],
#                                                              u_face_values[i, f],
#                                                              face_normal, face_normal,
#                                                              equations)

#             flux_face_values[i, f] = (cons_flux_at_face_node +
#                                       0.5 * noncons_flux_at_face_node) * Jf[i, f]
#         end
#     end

#     # Note: modifying the values of the reshaped array modifies the values of cache.flux_face_values.
#     # However, we don't have to re-reshape, since cache.flux_face_values still retains its original shape.
# end

# Multiple calc_sources! to resolve method ambiguities
function calc_sources!(du, u, t, source_terms::Nothing,
                       domain, equations, solver::PointCloudSolver, cache)
    nothing
end

# Redefined to allow for generic sources include sources 
# requiring operator application. Each source will
# be a callable struct containing its own caches
function calc_sources!(du, u, t, source_terms,
                       domain, equations, solver::PointCloudSolver, cache)
    for source in values(source_terms)
        save_tag = string(nameof(typeof(source)))
        @trixi_timeit timer() "calc $save_tag" source(du, u, t, domain, equations,
                                                      solver, cache)
    end
end

function Trixi.rhs!(du, u, t, domain, equations,
                    initial_condition, boundary_conditions::BC, source_terms::Source,
                    solver::PointCloudSolver, cache) where {BC, Source}
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
