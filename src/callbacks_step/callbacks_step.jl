####################################################################################################
# Include files with actual implementations for callbacks called after timestep is complete 

# include("residual_monitor.jl")
# include("save_solition.jl")

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Used by `SolutionSavingCallback` and `DensityReinitializationCallback`
get_iter(::Integer, integrator) = integrator.stats.naccept
function get_iter(dt::AbstractFloat, integrator)
    # Basically `(t - tspan[1]) / dt` as `Int`.
    Int(div(integrator.t - first(integrator.sol.prob.tspan), dt, RoundNearest))
end

# Used by `InfoCallback` and `PostProcessCallback`
@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
           isempty(integrator.opts.tstops) ||
           integrator.iter == integrator.opts.maxiters
end

@inline function condition_integrator_interval(integrator, interval;
                                               save_final_solution = true)
    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && (((integrator.stats.naccept % interval == 0) &&
             !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
            (save_final_solution && isfinished(integrator)))
end

# `include` callback definitions in the order that we currently prefer
# when combining them into a `CallbackSet` which is called *after* a complete step
# The motivation is as follows: The first callbacks belong to the current time step iteration:
# * `SummaryCallback` controls, among other things, timers and should thus be first
# * `SteadyStateCallback` may mark a time step as the last step, which is needed by other callbacks
# * `AnalysisCallback` may also do some checks that mark a step as the last one
# * `AliveCallback` belongs to `AnalysisCallback` and should thus be nearby
# * `SaveRestartCallback`, `SaveSolutionCallback`, and `TimeSeriesCallback` should save the current
#    solution state before it is potentially degraded by AMR
# * `VisualizationCallback` similarly should be called before the mesh is adapted
#
# From here on, the remaining callbacks essentially already belong to the next time step iteration:
# * `AMRCallback` really belongs to the next time step already, as it should be the "first" callback
#   in a time step loop (however, callbacks are always executed *after* a step, thus it comes near
#   the end here)
# * `StepsizeCallback` must come after AMR to accommodate potential changes in the minimum cell size
# * `GlmSpeedCallback` must come after computing time step size because it affects the value of c_h
# * `LBMCollisionCallback` must come after computing time step size because it is already part of
#    the next time step calculation
include("history.jl")
include("info.jl")
include("save_solution_vtk.jl")
end # @muladd
