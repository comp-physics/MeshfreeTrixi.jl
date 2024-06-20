# Based on Trixi/src/solvers/DGMulti/dg.jl
# Rewritten to work with RBF-FD methods on PointCloudDomains.
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Constructs cache variables for ParallelPointCloudDomains
# So far no changes from regular PointCloudSolver cache
function Trixi.create_cache(domain::ParallelPointCloudDomain{NDIMS}, equations,
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

function update_halos!(du, u, cache, t, boundary_conditions,
                       domain::ParallelPointCloudDomain,
                       have_nonconservative_terms, equations,
                       solver::PointCloudSolver)

    # Unpack required MPI information
    mpi_cache = domain.mpi_cache
    pd = domain.pd
    @unpack mpi_send_id, mpi_recv_id, mpi_send_idx, mpi_recv_length, mpi_send_buffers, mpi_recv_buffers, mpi_requests, n_elements_global, n_elements_local = mpi_cache

    # Perform p2p communication for halo regions
    u_local = @view(u[1:n_elements_local, :])
    halo = @view(u[(n_elements_local + 1):end, :])
    u_local, halo = perform_halo_update!(u_local, halo, mpi_send_id, mpi_recv_id,
                                         mpi_send_idx,
                                         mpi_recv_length, comm)
end

function Trixi.rhs!(du, u, t, domain::ParallelPointCloudDomain, equations,
                    initial_condition, boundary_conditions::BC, source_terms::Source,
                    solver::PointCloudSolver, cache) where {BC, Source}
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, solver, cache)

    @trixi_timeit timer() "update halos" begin
        update_halos!(du, u, cache, t, boundary_conditions, domain,
                      have_nonconservative_terms(equations), equations, solver)
    end

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
