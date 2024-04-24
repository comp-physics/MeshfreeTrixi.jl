"""
    SourceIGR

A struct containing everything needed to an IGR source for 
stabilization of RBF-FD discretizations of conservation laws.
Applies flux term 
drho_v1 -= Dx Σ
drho_v2 -= Dy Σ
Based on
(ρ⁻¹ - α (Dx ρ⁻¹ Dx + Dy ρ⁻¹ Dy)) Σ = α(tr([Du])² + tr([Du]²))
## References

- Cao, Schafer (2023)
  Information geometric regularization of the barotropic Euler equation
  [doi: 10.48550/arXiv.2308.14127](https://doi.org/10.48550/arXiv.2308.14127)
"""
struct SourceIGR{Cache}
    cache::Cache

    function SourceIGR{Cache}(cache::Cache) where {Cache}
        new(cache)
    end
end

"""
SourceIGR(solver, equations, domain)

Construct an Information Geometric Regularization source for an RBF-FD discretization.
"""
function SourceIGR(solver, equations, domain; alpha = 1.0, linear_solver = cg)
    cache = (; create_igr_cache(solver, equations, domain, alpha, linear_solver)...)

    SourceIGR{typeof(cache)}(cache)
end

struct IGRCompositeMatrix{T}
    Dx::SparseMatrixCSC{T, Int}
    Dy::SparseMatrixCSC{T, Int}
    rho_inv::Vector{T}
    alpha::T # 20(∆x)2 but passed in for now
    tmp1::Vector{T}
    tmp2::Vector{T}
    tmp3::Vector{T} # TODO: Make this nicer and generic
end
function *(A::IGRCompositeMatrix{T}, v::AbstractVector{T}) where {T}
    # Implements  (ρ⁻¹ - α (Dx ρ⁻¹ Dx + Dy ρ⁻¹ Dy))
    Dx_v = A.Dx * v
    Dy_v = A.Dy * v
    Dx_rho_inv_Dx_v = A.Dx * (A.rho_inv .* Dx_v)
    Dy_rho_inv_Dy_v = A.Dy * (A.rho_inv .* Dy_v)
    # Compute the full operation
    return A.rho_inv .* v - A.alpha * (Dx_rho_inv_Dx_v + Dy_rho_inv_Dy_v)
end
function mul!(y::AbstractVector{T}, A::IGRCompositeMatrix{T},
              v::AbstractVector{T}) where {T}
    # Use tmp1 for Dx_v and Dy_v
    mul!(A.tmp1, A.Dx, v)  # Dx_v in-place
    mul!(A.tmp2, A.Dy, v)  # Dy_v in-place

    # Apply rho_inv element-wise to tmp1 and tmp2, then perform Dx and Dy
    A.tmp1 .= A.rho_inv .* A.tmp1
    A.tmp2 .= A.rho_inv .* A.tmp2
    mul!(A.tmp3, A.Dx, A.tmp1)  # Dx_rho_inv_Dx_v in-place
    mul!(A.tmp1, A.Dy, A.tmp2)  # Reuse tmp1 for Dy_rho_inv_Dy_v in-place

    # Combine all parts to compute the final result in y
    y .= A.rho_inv .* v
    y .-= A.alpha .* (A.tmp3 .+ A.tmp1)  # Use .-= to subtract in-place
end
Base.eltype(A::IGRCompositeMatrix{T}) where {T} = T
Base.size(A::IGRCompositeMatrix, dim::Int) = size(A.Dx, dim)
Base.size(A::IGRCompositeMatrix) = (size(A.Dx, 1), size(A.Dx, 2))

function create_igr_cache(solver::PointCloudSolver, equations,
                          domain::PointCloudDomain, alpha, linear_solver)
    # Get basis and domain info
    basis = solver.basis
    pd = domain.pd

    # Create the actual operators
    # hv_differentiation_matrices operator
    # k is order of Laplacian, actual div order is 2k
    # rbf_differentiation_matrices = compute_flux_operator(solver, domain)
    # Can just reuse rbf_differentiation_matrices in update

    # Containers
    nvars = nvariables(equations)
    uEltype = real(solver)
    sigma = zeros(uEltype, pd.num_points)
    rho_inv = zeros(uEltype, pd.num_points)
    igr_rhs = zeros(uEltype, pd.num_points)
    trace = zeros(uEltype, pd.num_points)
    trace_squared = zeros(uEltype, pd.num_points)
    flux_x = allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)
    flux_y = allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)

    # Actual operators for 
    # Dx ρ⁻¹ Dx + Dy ρ⁻¹ Dy
    # tr([Du])² + tr([Du]²) 
    # where Du = [Dx⋅u_x Dy⋅u_x; 
    #             Dx⋅u_y Dy⋅u_y]
    # already exist in semi.cache.rbf_differentiation_matrices
    # Regenerating here for ease of implementation 
    # but could be improved
    tmp1 = zeros(uEltype, pd.num_points)
    tmp2 = zeros(uEltype, pd.num_points)
    tmp3 = zeros(uEltype, pd.num_points)
    differentiation_matrices = compute_flux_operator(solver, domain, 1)
    lhs_operator = IGRCompositeMatrix(differentiation_matrices[1],
                                      differentiation_matrices[2],
                                      rho_inv, alpha, tmp1, tmp2, tmp3)

    return (; sigma, rho_inv, igr_rhs, flux_x, flux_y, trace, trace_squared, alpha,
            lhs_operator, linear_solver)
end

function update_igr_rhs!(du, u,
                         equations::CompressibleEulerEquations2D, domain, cache,
                         semi_cache)
    # Calculate α(tr([Du])² + tr([Du]²))
    @unpack sigma, flux_x, flux_y, igr_rhs, trace, trace_squared, alpha, lhs_operator = cache
    @unpack u_values, local_values_threaded, rhs_local_threaded, rbf_differentiation_matrices = semi_cache
    u_prim = local_values_threaded[1]
    local_rhs = rhs_local_threaded[1]
    set_to_zero!(u_prim)
    set_to_zero!(local_rhs)
    set_to_zero!(flux_x)
    set_to_zero!(flux_y)

    for i in eachindex(du)
        u_prim[i] = cons2prim(u[i], equations)
    end

    apply_to_each_field(mul_by!(rbf_differentiation_matrices[1]),
                        flux_x, u_prim)
    apply_to_each_field(mul_by!(rbf_differentiation_matrices[2]),
                        flux_y, u_prim)

    # Calculate α(tr([Du])² + tr([Du]²))
    # TODO: Improve allocations
    for i in eachindex(igr_rhs)
        # tr([Du])²
        trace[i] = 0.0
        trace_squared[i] = 0.0
        trace[i] += flux_x[i][1]
        trace[i] += flux_y[i][2]
        # trace[i] .= trace[i] * trace[i]

        # tr([Du]²)
        # dx_ux_squared = flux_x[i][2]^2  # (∂ux/∂x)^2
        # dy_uy_squared = flux_y[i][3]^2  # (∂uy/∂y)^2
        # cross_term = 2 * flux_y[2][i] * flux_x[3][i]  # 2*(∂ux/∂y)*(∂uy/∂x)
        trace_squared[i] = flux_x[i][2]^2 +
                           2 * flux_y[i][2] * flux_x[i][3] +
                           flux_y[i][3]^2

        igr_rhs[i] = alpha * (trace[i] * trace[i] + trace_squared[i])
    end
end

function update_sigma!(sigma, du, u,
                       equations::CompressibleEulerEquations2D, domain, cache,
                       semi_cache)
    # Solve (ρ⁻¹ - α (Dx ρ⁻¹ Dx + Dy ρ⁻¹ Dy)) Σ = α(tr([Du])² + tr([Du]²))
    @unpack sigma, trace, trace_squared, alpha, lhs_operator, igr_rhs, linear_solver = cache
    @unpack u_values, local_values_threaded, rhs_local_threaded = semi_cache
    local_u = local_values_threaded[1]
    local_rhs = rhs_local_threaded[1]
    set_to_zero!(local_u)
    set_to_zero!(local_rhs)

    # set_to_zero!(sigma)
    sigma .= 0.0
    # sigma .= igr_rhs
    for i in eachindex(lhs_operator.rho_inv)
        lhs_operator.rho_inv[i] = 1.0 / u[i][1]
    end

    # Perform iterative solve here
    # Make solver a passable parameter
    linear_solver(sigma, lhs_operator, igr_rhs; maxiter = 20)
end

# Calculate 2D flux for a single point
@inline function flux_igr(sigma, orientation::Integer,
                          equations::CompressibleEulerEquations2D)
    if orientation == 1
        f1 = 0.0
        f2 = sigma
        f3 = 0.0
        f4 = 0.0
    else
        f1 = 0.0
        f2 = 0.0
        f3 = sigma
        f4 = 0.0
    end
    return SVector(f1, f2, f3, f4)
end

function (source::SourceIGR)(du, u, t, domain, equations,
                             solver::PointCloudSolver, semi_cache)
    basis = solver.basis
    pd = domain.pd
    @unpack sigma, trace, trace_squared, alpha, lhs_operator = source.cache
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded, rhs_local_threaded = semi_cache

    # Update rhs, lhs system, and solve for sigma
    # For now, lhs will likely allocate
    update_igr_rhs!(du, u, equations, domain, source.cache, semi_cache)
    update_sigma!(sigma, du, u, equations, domain, source.cache, semi_cache)
    # update_visc!(eps, eps_c, eps_uw, sigma, source.cache.success_iter[1])

    # Compute the hyperviscous dissipation
    # We need to apply P ⋅ u = Dx' diag(eps) Dx u + Dy' diag(eps) Dy u + ...
    # Handle one dim at a time, then accumulate into du
    flux_values = local_values_threaded[1]
    set_to_zero!(flux_values)
    # Applies flux terms
    # drho_v1 -= Dx Σ
    # drho_v2 -= Dy Σ
    for i in eachdim(domain)
        for e in eachelement(domain, solver, semi_cache)
            flux_values[e] = flux_igr(sigma[e], i, equations)
        end
        apply_to_each_field(mul_by_accum!(rbf_differentiation_matrices[i],
                                          -1),
                            du, flux_values)
    end
    # du .+= local_rhs
end
