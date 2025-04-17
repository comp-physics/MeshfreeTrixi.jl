# Based on Trixi/src/callbacks_stage/positivity_zhang_shu.jl
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# @inline function update_mean(u_mean, u_next,
#                              equations::CompressibleEulerEquations2D)
#     # u_mean[element] += u[i]
#     rho_m, rho_v1_m, rho_v2_m, rho_e_m = u_mean
#     rho, rho_v1, rho_v2, rho_e = u_next
#     rho_m += rho
#     rho_v1_m += rho_v1
#     rho_v2_m += rho_v2
#     rho_e_m += rho_e
#     return SVector(rho_m, rho_v1_m, rho_v2_m, rho_e_m)
# end

"""
    PositivityPreservingLimiterZhangShu(; threshold, variables)

The fully-discrete positivity-preserving limiter of
- Zhang, Shu (2011)
    Maximum-principle-satisfying and positivity-preserving high-order schemes
    for conservation laws: survey and new developments
    [doi: 10.1098/rspa.2011.0153](https://doi.org/10.1098/rspa.2011.0153)
The limiter is applied to all scalar `variables` in their given order
using the associated `thresholds` to determine the minimal acceptable values.
The order of the `variables` is important and might have a strong influence
on the robustness.
"""
function Trixi.limiter_zhang_shu!(u, threshold::Real, variable,
                                  domain::PointCloudDomain{2}, equations,
                                  solver::PointCloudSolver, cache)
    # @unpack weights = solver.basis
    local_u = cache.local_values_threaded[1]
    u_mean = cache.rhs_local_threaded[1]
    set_to_zero!(local_u)
    set_to_zero!(u_mean)
    zero_el = SVector(zeros(eltype(u[1]), nvariables(equations))...)

    # @threaded for element in eachelement(solver, cache)
    for element in eachindex(u)
        # determine minimum value
        value_min = typemax(eltype(u[element]))
        # neighbors_included = min(7, domain.pd.num_neighbors)
        neighbors_included = domain.pd.num_neighbors
        for i in domain.pd.neighbors[element][1:neighbors_included]
            value_min = min(value_min, variable(u[i], equations))
        end
        value_min = min(value_min, variable(u[element], equations))

        # detect if limiting is necessary
        value_min < threshold || continue

        # compute mean value
        update_mean(u_mean[element], u, element,
                    equations::CompressibleEulerEquations2D, domain)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean[element], equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        # local_u[element] = theta * u[element] + (1 - theta) * u_mean[element]
        apply_limiter!(local_u[element], u[element], u_mean[element], theta)
    end

    # Apply limited values
    for element in eachindex(u)
        if local_u[element] != zero_el
            u[element] = local_u[element]
        end
    end

    return nothing
end
@inline function update_mean(u_mean, u, element,
                             equations::CompressibleEulerEquations2D, domain)
    rho_m, rho_v1_m, rho_v2_m, rho_e_m = u_mean
    # neighbors_included = min(7, domain.pd.num_neighbors)
    neighbors_included = domain.pd.num_neighbors
    for i in domain.pd.neighbors[element][1:neighbors_included]
        rho, rho_v1, rho_v2, rho_e = u[i]
        rho_m += rho
        rho_v1_m += rho_v1
        rho_v2_m += rho_v2
        rho_e_m += rho_e
    end
    # u_mean[element] = u_mean[element] / domain.pd.num_neighbors
    rho_m = rho_m / neighbors_included
    rho_v1_m = rho_v1_m / neighbors_included
    rho_v2_m = rho_v2_m / neighbors_included
    rho_e_m = rho_e_m / neighbors_included
    return SVector(rho_m, rho_v1_m, rho_v2_m, rho_e_m)
end
@inline function apply_limiter!(local_u, u, u_mean, theta)
    # local_u[element] = theta * u[element] + (1 - theta) * u_mean[element]
    rho, rho_v1, rho_v2, rho_e = u
    rho_m, rho_v1_m, rho_v2_m, rho_e_m = u_mean
    rho_l, rho_v1_l, rho_v2_l, rho_e_l = local_u

    rho_l = theta * rho + (1 - theta) * rho_m
    rho_v1_l = theta * rho_v1 + (1 - theta) * rho_v1_m
    rho_v2_l = theta * rho_v2 + (1 - theta) * rho_v2_m
    rho_e_l = theta * rho_e + (1 - theta) * rho_e_m

    return SVector(rho_l, rho_v1_l, rho_v2_l, rho_e_l)
end

function Trixi.limiter_zhang_shu!(u, threshold::Real, variable,
                                  domain::PointCloudDomain{2}, equations,
                                  solver::CUDAPointCloudSolver, cache)
    threads = 256
    numblocks = ceil(Int, size(u)[1] / threads)

    # Prepare scratch space
    # @unpack weights = solver.basis
    local_u = cache.local_values_threaded[1]
    u_mean = cache.rhs_local_threaded[1]
    cache.flux_face_values .= 0.0
    u_var = @view cache.flux_face_values[:, 1]
    u_min = @view cache.flux_face_values[:, 2]
    u_mean_var = @view cache.flux_face_values[:, 3]
    theta = @view cache.flux_face_values[:, 4]
    d_neighbors = cache.d_neighbors
    d_u_var_neighbors = cache.d_u_var_neighbors
    d_threshold_true = cache.d_threshold_true
    # set_to_zero!(local_u)
    # set_to_zero!(u_mean)
    local_u .= 0.0
    u_mean .= 0.0
    # u_var .= 0.0
    # u_min .= 0.0
    # zero_el = SVector(zeros(eltype(u[1]), nvariables(equations))...)

    ### Get variable first
    # @device_code_warntype interactive=true 
    @cuda threads=threads blocks=numblocks variable_kernel!(u_var, u, variable,
                                                            equations)

    ### Then find min in neighbors with array subindexing 
    # @cuda threads=threads blocks=numblocks local_min_kernel!(u_min, u_var, u, variable, equations) ### Issue here is minimum inside of kernel, would need to manually accumulate
    d_u_var_neighbors .= u_var[d_neighbors]
    u_min .= minimum(d_u_var_neighbors, dims = 1)'
    ### Then create bitarray where u_min < threshold
    d_threshold_true .= u_min[:] .< threshold
    # if any(d_threshold_true)  

    # Calculate u_mean
    for i in 1:size(u)[2]
        d_u_var_neighbors .= u[:, i][d_neighbors]
        u_mean[:, i] .= mean(d_u_var_neighbors, dims = 1)'
    end

    # Calculate theta
    theta .= (u_mean_var .- threshold) ./ (u_mean_var .- u_min)
    local_u .= theta .* u .+ (1 .- theta) .* u_mean

    # Apply local_u to u when threshold == true
    u[d_threshold_true[:], :] .= local_u[d_threshold_true[:], :]
end
function variable_kernel!(u_var, u, variable, equations)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for element in index:stride:size(u)[1]
        u_var[element] = variable_cuda(@view(u[element, :]), variable, equations)
    end
    return nothing
end
function variable_cuda(u::U, variable::F, equations::E) where {U, F, E}
    return variable(u, equations)
end
# function local_min_kernel!(u_min, u_var, u, d_neighbors)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     # @threaded for element in eachelement(solver, cache)
#     for element in index:stride:size(u)[1]
#         ### Serial loop will not work with GPU without atomic
#         ### Need to vectorize the calculation
#         # determine minimum value
#         # for i in domain.pd.neighbors[element][1:min(7, end)]
#         #     value_min = min(value_min, variable(u[i], equations))
#         # end
#         # neighbors = (num_neighbors*(element-1)+1):(num_neighbors*(element-1)+num_neighbors)
#         value_min = minimum(u_var[d_neighbors[:, element]])
#     end
#     return nothing
# end
# function limiter_zhang_shu_kernel!(local_u, u_var, u, u_mean, d_neighbors, variable,
#                                     threshold, equations)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     # @threaded for element in eachelement(solver, cache)
#     for element in index:stride:size(u)[1]
#         ### Serial loop will not work with GPU without atomic
#         ### Need to vectorize the calculation
#         # determine minimum value
#         # for i in domain.pd.neighbors[element][1:min(7, end)]
#         #     value_min = min(value_min, variable(u[i], equations))
#         # end
#         neighbors = (num_neighbors * (element - 1) + 1):(num_neighbors * (element - 1) + num_neighbors)
#         value_min = minimum(u_var[d_neighbors[neighbors]])

#         # detect if limiting is necessary
#         value_min < threshold || continue

#         # compute mean value
#         update_mean_kernel(u_mean[element, :], u, element,
#                             equations::CompressibleEulerEquations2D, domain)

#         # We compute the value directly with the mean values, as we assume that
#         # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
#         value_mean = variable(u_mean[element, :], equations)
#         theta = (value_mean - threshold) / (value_mean - value_min)
#         # local_u[element] = theta * u[element] + (1 - theta) * u_mean[element]
#         apply_limiter_kernel(local_u[element, :], u[element, :], u_mean[element, :],
#                                 theta)
#     end
#     return nothing
# end
# function apply_limiter_zhang_shu_kernel!(u, local_u)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     # Apply limited values
#     for element in index:stride:size(u)[1]
#         if any(local_u[element, :] .!= 0.0)
#             u[element, :] = local_u[element, :]
#         end
#     end
#     return nothing
# end
# function update_mean_kernel(u_mean::UM, u::U, element::E,
#                             equations::CompressibleEulerEquations2D,
#                             domain::D) where {UM, U, E, D}
#     update_mean(u_mean, u, element,
#                 equations::CompressibleEulerEquations2D, domain)
# end
# function apply_limiter_kernel(local_u::LU, u::U, u_mean::UM,
#                                 theta::T) where {LU, U, UM, T}
#     apply_limiter!(local_u, u, u_mean, theta)
# end
end # @muladd
