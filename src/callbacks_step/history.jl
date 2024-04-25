# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    HistoryCallback(; approx_order=1.0)

Update solution time history according to a spline interpolation with 
reconstruction order `approx_order`. Utilized to approximate residual 
for targeted residual based viscosity.
"""
mutable struct HistoryCallback{RealT}
    approx_order::RealT
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:HistoryCallback})
    @nospecialize cb # reduce precompilation time

    history_callback = cb.affect!
    @unpack approx_order = history_callback
    print(io, "HistoryCallback(approx_order=", approx_order, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:HistoryCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        history_callback = cb.affect!

        setup = [
            "Reconstruction Order" => history_callback.approx_order
        ]
        summary_box(io, "HistoryCallback", setup)
    end
end

function HistoryCallback(; approx_order::Int)
    history_callback = HistoryCallback(approx_order)

    DiscreteCallback(history_callback, history_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: HistoryCallback}
    cb.affect!(integrator)
end

# this method is called to determine whether the callback should be activated
function (history_callback::HistoryCallback)(u, t, integrator)
    return true
end

# This method is called as callback during the time integration.
@inline function (history_callback::HistoryCallback)(integrator)
    t = integrator.t
    u_ode = integrator.u
    semi = integrator.p
    @unpack approx_order = history_callback

    # Dispatch based on semidiscretization
    @trixi_timeit timer() "update history" update_history!(semi, u_ode, t, approx_order,
                                                           integrator)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

function update_history!(semi, u, t, approx_order, integrator)
    @unpack source_terms = semi

    # If source includes time history cache, update
    # otherwise no-op
    for source in values(source_terms)
        modify_cache!(source, u, t, approx_order, integrator)
    end
end

function modify_cache!(source::T, u, t, approx_order, integrator) where {T}
    # Fallback method that does nothing
end

function modify_cache!(source::SourceResidualViscosityTominec, u, t, approx_order,
                       integrator)
    # May need access to integrator to get count of timesteps 
    # since first few iterations can only support lower order
    @unpack time_history, sol_history, approx_du, time_weights = source.cache
    @unpack success_iter, iter, saveiter, saveiter_dense, last_stepfail, accept_step = integrator

    source.cache.success_iter .= integrator.success_iter

    shift_soln_history!(time_history, sol_history, t, u)
    update_approx_du!(approx_du, time_weights, time_history, sol_history, success_iter,
                      approx_order)
end

function shift_soln_history!(time_history, sol_history, t, u)
    # Assuming sol_history[:, 1] is the most recent and sol_history[:, end] is the oldest
    time_history[2:end] .= time_history[1:(end - 1)]
    time_history[1] = t
    sol_history[:, 2:end] .= sol_history[:, 1:(end - 1)]
    sol_history[:, 1] .= u
end

function update_approx_du!(approx_du, time_weights, time_history, sol_history,
                           success_iter, approx_order)
    set_to_zero!(approx_du)

    num_time_points = min(success_iter + 1, approx_order + 1)
    if success_iter > 0
        # Update the time weights for the current number of time points
        time_deriv_weights!(@view(time_weights[1:num_time_points]),
                            @view(time_history[1:num_time_points]))

        for i in 1:num_time_points
            approx_du .+= time_weights[i] .* sol_history[:, i]
        end
    end

    return nothing
end

function time_deriv_weights!(w, t)
    #Input: a vector t, where t(i) is time at which the solution is available.
    # Output: a vector w, where each w(i) is used to multiply u|_{t(i)} in order
    # ... to get a derivative at t(end).
    # Usage: d/dt u(t_end) = w(end)*u(end) + w(end-1)*u(end-1) + ... + w(1)*u(1),
    # ... where t_end is the time at which the last solution point is available.
    # From Tominec
    scale = 1 / maximum(abs.(t))
    t_ = t .* scale
    t_eval = t_[1] # The derivative should be evaluated at t(end).
    # Construct the polynomial basis, and differentiate it in a point t_eval.
    A = zeros(size(t_, 1), size(t_, 1))
    b_t = zeros(1, size(t_, 1))
    for k in 1:length(t)
        A[:, k] = t_ .^ (k - 1)
        b_t[k] = (k - 1) * t_eval .^ (k - 2)
    end
    # w .= scale .* (b_t / A)
    w .= scale .* (A' \ b_t')

    return nothing
end
end # @muladd
