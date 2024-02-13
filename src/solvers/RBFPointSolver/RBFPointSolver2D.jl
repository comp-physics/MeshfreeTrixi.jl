function rhs!(du, u, t,
              mesh::PointCloudDomain, equations,
              initial_condition, boundary_conditions, source_terms::Source,
              dg::PointCloudSolver, cache) where {Source}
    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    # @trixi_timeit timer() "volume integral" begin
    # calc_volume_integral!(du, u, mesh,
    #                     have_nonconservative_terms(equations), equations,
    #                     dg.volume_integral, dg, cache)
    # end

    # Prolong solution to interfaces
    # @trixi_timeit timer() "prolong2interfaces" begin
    # prolong2interfaces!(cache, u, mesh, equations,
    #                   dg.surface_integral, dg)
    # end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    # @trixi_timeit timer() "prolong2boundaries" begin
    # prolong2boundaries!(cache, u, mesh, equations,
    #                   dg.surface_integral, dg)
    # end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Prolong solution to mortars
    # @trixi_timeit timer() "prolong2mortars" begin
    # prolong2mortars!(cache, u, mesh, equations,
    #                dg.mortar, dg.surface_integral, dg)
    # end

    # Calculate mortar fluxes
    # @trixi_timeit timer() "mortar flux" begin
    # calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
    #                 have_nonconservative_terms(equations), equations,
    #                 dg.mortar, dg.surface_integral, dg, cache)
    # end

    # Calculate surface integrals
    # @trixi_timeit timer() "surface integral" begin
    # calc_surface_integral!(du, u, mesh, equations,
    #                      dg.surface_integral, dg, cache)
    # end

    # Calculate surface fluxes (i.e. normal fluxes)
    @trixi_timeit timer() "surface flux" begin
        calc_surface_flux!(du, u, mesh, equations,
                           dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    # @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache)
    end

    return nothing
end