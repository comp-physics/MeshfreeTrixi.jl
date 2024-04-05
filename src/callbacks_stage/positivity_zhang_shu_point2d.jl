# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function Trixi.limiter_zhang_shu!(u, threshold::Real, variable,
                                  domain::PointCloudDomain{2}, equations,
                                  solver::PointCloudSolver, cache)
    # @unpack weights = solver.basis
    local_u = cache.local_values_threaded[1]
    set_to_zero!(local_u)

    # @threaded for element in eachelement(solver, cache)
    for element in eachindex(u)
        # determine minimum value
        value_min = typemax(eltype(u[element]))
        # u_node = u[domain.pd.neighbors[element]]
        # value_min = min(value_min, minimum(u_node))
        # variable(u[element], equations)
        for i in domain.pd.neighbors[element]
            value_min = min(value_min, variable(u[i], equations))
        end
        # for j in eachnode(solver), i in eachnode(solver)
        #     u_node = get_node_vars(u, equations, solver, i, j, element)
        #     value_min = min(value_min, variable(u_node, equations))
        # end

        # detect if limiting is necessary
        value_min < threshold || continue

        # compute mean value
        u_mean = SVector(zeros(eltype(u[element]), nvariables(equations))...)
        for i in domain.pd.neighbors[element]
            u_mean += u[i]
        end
        u_mean = u_mean / domain.pd.num_neighbors
        # for j in eachnode(solver), i in eachnode(solver)
        #     u_node = get_node_vars(u, equations, solver, i, j, element)
        #     u_mean += u_node * weights[i] * weights[j]
        # end
        # note that the reference element is [-1,1]^ndims(solver), thus the weights sum to 2
        # u_mean = u_mean / 2^ndims(domain)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        local_u[element] = theta * u[element] + (1 - theta) * u_mean
        # for j in eachnode(solver), i in eachnode(solver)
        #     u_node = get_node_vars(u, equations, solver, i, j, element)
        #     set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
        #                    equations, solver, i, j, element)
        # end
    end

    zero_el = SVector(zeros(eltype(u[1]), nvariables(equations))...)
    for element in eachindex(u)
        if local_u[element] != zero_el
            u[element] = local_u[element]
        end
    end

    return nothing
end
end # @muladd
