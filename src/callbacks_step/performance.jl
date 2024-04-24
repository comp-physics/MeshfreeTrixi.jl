# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# TODO: Taal refactor
# - analysis_interval part as PeriodicCallback called after a certain amount of simulation time
"""
    PerformanceCallback(semi; interval=0,
                           save_analysis=false,
                           output_directory="out",
                           analysis_filename="analysis.dat",
                           extra_analysis_errors=Symbol[],
                           extra_analysis_integrals=())

Analyze a numerical solution every `interval` time steps and print the
results to the screen. If `save_analysis`, the results are also saved in
`joinpath(output_directory, analysis_filename)`.

Additional errors can be computed, e.g. by passing
`extra_analysis_errors = (:l2_error_primitive, :linf_error_primitive)`
or `extra_analysis_errors = (:conservation_error,)`.

If you want to omit the computation (to safe compute-time) of the [`default_analysis_errors`](@ref), specify
`analysis_errors = Symbol[]`.
Note: `default_analysis_errors` are `:l2_error` and `:linf_error` for all equations.
If you want to compute `extra_analysis_errors` such as `:conservation_error` solely, i.e., 
without `:l2_error, :linf_error` you need to specify 
`analysis_errors = [:conservation_error]` instead of `extra_analysis_errors = [:conservation_error]`.

Further scalar functions `func` in `extra_analysis_integrals` are applied to the numerical
solution and integrated over the computational domain. Some examples for this are
[`entropy`](@ref), [`energy_kinetic`](@ref), [`energy_internal`](@ref), and [`energy_total`](@ref).
You can also write your own function with the same signature as the examples listed above and
pass it via `extra_analysis_integrals`.
See the developer comments about `Trixi.analyze`, `Trixi.pretty_form_utf`, and
`Trixi.pretty_form_ascii` for further information on how to create custom analysis quantities.

In addition, the analysis callback records and outputs a number of quantities that are useful for
evaluating the computational performance, such as the total runtime, the performance index
(time/DOF/rhs!), the time spent in garbage collection (GC), or the current memory usage (alloc'd
memory).
"""
mutable struct PerformanceCallback{Analyzer, AnalysisIntegrals, InitialStateIntegrals,
                                   Cache}
    start_time::Float64
    start_time_last_analysis::Float64
    ncalls_rhs_last_analysis::Int
    start_gc_time::Float64
    interval::Int
    save_analysis::Bool
    output_directory::String
    analysis_filename::String
    analyzer::Analyzer
    analysis_errors::Vector{Symbol}
    analysis_integrals::AnalysisIntegrals
    initial_state_integrals::InitialStateIntegrals
    cache::Cache
end

# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, performance_callback::PerformanceCallback)
# end
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:PerformanceCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        performance_callback = cb.affect!

        setup = Pair{String, Any}["interval" => performance_callback.interval,
                                  "analyzer" => performance_callback.analyzer]
        for (idx, error) in enumerate(performance_callback.analysis_errors)
            push!(setup, "│ error " * string(idx) => error)
        end
        for (idx, integral) in enumerate(performance_callback.analysis_integrals)
            push!(setup, "│ integral " * string(idx) => integral)
        end
        push!(setup,
              "save analysis to file" => performance_callback.save_analysis ? "yes" :
                                         "no")
        if performance_callback.save_analysis
            push!(setup, "│ filename" => performance_callback.analysis_filename)
            push!(setup,
                  "│ output directory" => abspath(normpath(performance_callback.output_directory)))
        end
        summary_box(io, "PerformanceCallback", setup)
    end
end

# This is the convenience constructor that gets called from the elixirs
function PerformanceCallback(semi::Trixi.AbstractSemidiscretization; kwargs...)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    PerformanceCallback(mesh, equations, solver, cache; kwargs...)
end

# This is the actual constructor
function PerformanceCallback(mesh, equations::Trixi.AbstractEquations, solver, cache;
                             interval = 0,
                             save_analysis = false,
                             output_directory = "out",
                             analysis_filename = "analysis.dat",
                             extra_analysis_errors = Symbol[],
                             analysis_errors = Symbol[],
                             extra_analysis_integrals = (),
                             analysis_integrals = Symbol[],
                             RealT = real(solver),
                             uEltype = eltype(cache.elements),
                             kwargs...)
    # Decide when the callback is activated.
    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    condition = (u, t, integrator) -> interval > 0 &&
        ((integrator.stats.naccept % interval == 0 &&
          !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
         isfinished(integrator))

    analyzer = SolutionAnalyzer(solver; kwargs...)
    cache_analysis = create_cache_analysis(analyzer, mesh, equations, solver, cache,
                                           RealT, uEltype)

    performance_callback = PerformanceCallback(0.0, 0.0, 0, 0.0,
                                               interval, save_analysis,
                                               output_directory,
                                               analysis_filename,
                                               analyzer,
                                               analysis_errors,
                                               Tuple(analysis_integrals),
                                               SVector(ntuple(_ -> zero(uEltype),
                                                              Val(nvariables(equations)))),
                                               cache_analysis)

    DiscreteCallback(condition, performance_callback,
                     save_positions = (false, false),
                     initialize = initialize!)
end

# This method gets called from OrdinaryDiffEq's `solve(...)`
function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t,
                     integrator) where {Condition, Affect! <: PerformanceCallback}
    semi = integrator.p
    du_ode = first(get_tmp_cache(integrator))
    initialize!(cb, u_ode, du_ode, t, integrator, semi)
end

# This is the actual initialization method
# Note: we have this indirection to allow initializing a callback from the PerformanceCallbackCoupled
function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, du_ode, t,
                     integrator, semi) where {Condition, Affect! <: PerformanceCallback}
    # initial_state_integrals = integrate(u_ode, semi)
    _, equations, _, _ = Trixi.mesh_equations_solver_cache(semi)

    performance_callback = cb.affect!
    # performance_callback.initial_state_integrals = initial_state_integrals
    @unpack save_analysis, output_directory, analysis_filename, analysis_errors, analysis_integrals = performance_callback

    if save_analysis
        mkpath(output_directory)

        # write header of output file
        open(joinpath(output_directory, analysis_filename), "w") do io
            @printf(io, "#%-8s", "timestep")
            @printf(io, "  %-14s", "time")
            @printf(io, "  %-14s", "dt")

            # for quantity in analysis_integrals
            #     @printf(io, "   %-14s", pretty_form_ascii(quantity))
            # end

            println(io)
        end
    end

    # Record current time using a high-resolution clock
    performance_callback.start_time = time_ns()

    # Record current time for performance index computation
    performance_callback.start_time_last_analysis = time_ns()

    # Record current number of `rhs!` calls for performance index computation
    performance_callback.ncalls_rhs_last_analysis = Trixi.ncalls(semi.performance_counter)

    # Record total time spent in garbage collection so far using a high-resolution clock
    # Note: For details see the actual callback function below
    performance_callback.start_gc_time = Base.gc_time_ns()

    performance_callback(u_ode, du_ode, integrator, semi)
    return nothing
end

# This method gets called from OrdinaryDiffEq's `solve(...)`
function (performance_callback::PerformanceCallback)(integrator)
    semi = integrator.p
    du_ode = first(get_tmp_cache(integrator))
    u_ode = integrator.u
    performance_callback(u_ode, du_ode, integrator, semi)
end

# This method gets called internally as the main entry point to the AnalysiCallback
# TODO: Taal refactor, allow passing an IO object (which could be devnull to avoid cluttering the console)
function (performance_callback::PerformanceCallback)(u_ode, du_ode, integrator, semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    @unpack dt, t = integrator
    iter = integrator.stats.naccept

    # Compute the percentage of the simulation that is done
    t = integrator.t
    t_initial = first(integrator.sol.prob.tspan)
    t_final = last(integrator.sol.prob.tspan)
    sim_time_percentage = (t - t_initial) / (t_final - t_initial) * 100

    # Record performance measurements and compute performance index (PID)
    runtime_since_last_analysis = 1.0e-9 * (time_ns() -
                                   performance_callback.start_time_last_analysis)
    # PID is an MPI-aware measure of how much time per global degree of freedom (i.e., over all ranks)
    # and per `rhs!` evaluation is required. MPI-aware means that it essentially adds up the time
    # spent on each MPI rank. Thus, in an ideally parallelized program, the PID should be constant
    # independent of the number of MPI ranks used, since, e.g., using 4x the number of ranks should
    # divide the runtime on each rank by 4. See also the Trixi.jl docs ("Performance" section) for
    # more information.
    ncalls_rhs_since_last_analysis = (Trixi.ncalls(semi.performance_counter)
                                      -
                                      performance_callback.ncalls_rhs_last_analysis)
    performance_index = runtime_since_last_analysis * 1 /
                        (ndofsglobal(mesh, solver, cache)
                         *
                         ncalls_rhs_since_last_analysis)
    # performance_index = runtime_since_last_analysis * mpi_nranks() /
    #                     (ndofsglobal(mesh, solver, cache)
    #                      *
    #                      ncalls_rhs_since_last_analysis)

    # Compute the total runtime since the analysis callback has been initialized, in seconds
    runtime_absolute = 1.0e-9 * (time_ns() - performance_callback.start_time)

    # Compute the relative runtime as time spent in `rhs!` divided by the number of calls to `rhs!`
    # and the number of local degrees of freedom
    # OBS! This computation must happen *after* the PID computation above, since `take!(...)`
    #      will reset the number of calls to `rhs!`
    runtime_relative = 1.0e-9 * take!(semi.performance_counter) / Trixi.ndofs(semi)

    # Compute the total time spent in garbage collection since the analysis callback has been
    # initialized, in seconds
    # Note: `Base.gc_time_ns()` is not part of the public Julia API but has been available at least
    #        since Julia 1.6. Should this function be removed without replacement in a future Julia
    #        release, just delete this analysis quantity from the callback.
    # Source: https://github.com/JuliaLang/julia/blob/b540315cb4bd91e6f3a3e4ab8129a58556947628/base/timing.jl#L83-L84
    gc_time_absolute = 1.0e-9 * (Base.gc_time_ns() - performance_callback.start_gc_time)

    # Compute the percentage of total time that was spent in garbage collection
    gc_time_percentage = gc_time_absolute / runtime_absolute * 100

    # Obtain the current memory usage of the Julia garbage collector, in MiB, i.e., the total size of
    # objects in memory that have been allocated by the JIT compiler or the user code.
    # Note: `Base.gc_live_bytes()` is not part of the public Julia API but has been available at least
    #        since Julia 1.6. Should this function be removed without replacement in a future Julia
    #        release, just delete this analysis quantity from the callback.
    # Source: https://github.com/JuliaLang/julia/blob/b540315cb4bd91e6f3a3e4ab8129a58556947628/base/timing.jl#L86-L97
    memory_use = Base.gc_live_bytes() / 2^20 # bytes -> MiB

    @trixi_timeit timer() "analyze solution" begin
        # General information
        println()
        println("─"^100)
        println(" Simulation running '", Trixi.get_name(equations), "' with ",
                summary(solver))
        println("─"^100)
        println(" #timesteps:     " * @sprintf("% 14d", iter) *
                "               " *
                " run time:       " * @sprintf("%10.8e s", runtime_absolute))
        println(" Δt:             " * @sprintf("%10.8e", dt) *
                "               " *
                " └── GC time:    " *
                @sprintf("%10.8e s (%5.3f%%)", gc_time_absolute, gc_time_percentage))
        println(rpad(" sim. time:      " *
                     @sprintf("%10.8e (%5.3f%%)", t, sim_time_percentage), 46) *
                " time/DOF/rhs!:  " * @sprintf("%10.8e s", runtime_relative))
        println("                 " * "              " *
                "               " *
                " PID:            " * @sprintf("%10.8e s", performance_index))
        println(" #DOFs per field:" * @sprintf("% 14d", Trixi.ndofs(semi)) *
                "               " *
                " alloc'd memory: " * @sprintf("%14.3f MiB", memory_use))
        println(" #elements:      " *
                @sprintf("% 14d", nelements(mesh, solver, cache)))

        # Level information (only show for AMR)
        # print_amr_information(integrator.opts.callback, mesh, solver, cache)
        # println()

        # Open file for appending and store time step and time information
        if performance_callback.save_analysis
            io = open(joinpath(performance_callback.output_directory,
                               performance_callback.analysis_filename), "a")
            @printf(io, "% 9d", iter)
            @printf(io, "  %10.8e", t)
            @printf(io, "  %10.8e", dt)
        else
            io = devnull
        end

        # Calculate current time derivative (needed for semidiscrete entropy time derivative, residual, etc.)
        # `integrator.f` is usually just a call to `rhs!`
        # However, we want to allow users to modify the ODE RHS outside of Trixi.jl
        # and allow us to pass a combined ODE RHS to OrdinaryDiffEq, e.g., for
        # hyperbolic-parabolic systems.
        # @notimeit timer() integrator.f(du_ode, u_ode, semi, t)
        # u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
        # du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)
        # Compute l2_error, linf_error
        # performance_callback(io, du, u, u_ode, t, semi)

        println("─"^100)
        println()

        flush(stdout)

        # Add line break and close analysis file if it was opened
        if performance_callback.save_analysis
            # This resolves a possible type instability introduced above, since `io`
            # can either be an `IOStream` or `devnull`, but we know that it must be
            # an `IOStream here`.
            println(io::IOStream)
            close(io::IOStream)
        end
    end

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)

    # Reset performance measurements
    performance_callback.start_time_last_analysis = time_ns()
    performance_callback.ncalls_rhs_last_analysis = Trixi.ncalls(semi.performance_counter)

    return nothing
end

# used for error checks and EOC analysis
function (cb::DiscreteCallback{Condition, Affect!})(sol) where {Condition,
                                                                Affect! <:
                                                                AnalysisCallback}
    analysis_callback = cb.affect!
    semi = sol.prob.p
    @unpack analyzer = analysis_callback
    cache_analysis = analysis_callback.cache

    # l2_error, linf_error = calc_error_norms(sol.u[end], sol.t[end], analyzer, semi,
    # cache_analysis)
    # (; l2 = l2_error, linf = linf_error)
end

# This method is just called internally from `(performance_callback::PerformanceCallback)(integrator)`
# and serves as a function barrier. Additionally, it makes the code easier to profile and optimize.
# function (performance_callback::PerformanceCallback)(io, du, u, u_ode, t, semi)
#     # We only care about the timings 
#     # @unpack analyzer, analysis_errors, analysis_integrals = performance_callback
#     # cache_analysis = performance_callback.cache
#     # _, equations, _, _ = mesh_equations_solver_cache(semi)

#     # # Calculate and print derived quantities (error norms, entropy etc.)
#     # # Variable names required for L2 error, Linf error, and conservation error
#     # if any(q in analysis_errors
#     #        for q in (:l2_error, :linf_error, :conservation_error, :residual)) &&
#     #    mpi_isroot()
#     #     print(" Variable:    ")
#     #     for v in eachvariable(equations)
#     #         @printf("   %-14s", varnames(cons2cons, equations)[v])
#     #     end
#     #     println()
#     # end

#     # if :l2_error in analysis_errors || :linf_error in analysis_errors
#     #     # Calculate L2/Linf errors
#     #     l2_error, linf_error = calc_error_norms(u_ode, t, analyzer, semi,
#     #                                             cache_analysis)

#     #     if mpi_isroot()
#     #         # L2 error
#     #         if :l2_error in analysis_errors
#     #             print(" L2 error:    ")
#     #             for v in eachvariable(equations)
#     #                 @printf("  % 10.8e", l2_error[v])
#     #                 @printf(io, "  % 10.8e", l2_error[v])
#     #             end
#     #             println()
#     #         end

#     #         # Linf error
#     #         if :linf_error in analysis_errors
#     #             print(" Linf error:  ")
#     #             for v in eachvariable(equations)
#     #                 @printf("  % 10.8e", linf_error[v])
#     #                 @printf(io, "  % 10.8e", linf_error[v])
#     #             end
#     #             println()
#     #         end
#     #     end
#     # end

#     # # Conservation error
#     # if :conservation_error in analysis_errors
#     #     @unpack initial_state_integrals = performance_callback
#     #     state_integrals = integrate(u_ode, semi)

#     #     if mpi_isroot()
#     #         print(" |∑U - ∑U₀|:  ")
#     #         for v in eachvariable(equations)
#     #             err = abs(state_integrals[v] - initial_state_integrals[v])
#     #             @printf("  % 10.8e", err)
#     #             @printf(io, "  % 10.8e", err)
#     #         end
#     #         println()
#     #     end
#     # end

#     # # Residual (defined here as the vector maximum of the absolute values of the time derivatives)
#     # if :residual in analysis_errors
#     #     mpi_print(" max(|Uₜ|):   ")
#     #     for v in eachvariable(equations)
#     #         # Calculate maximum absolute value of Uₜ
#     #         res = maximum(abs, view(du, v, ..))
#     #         if mpi_isparallel()
#     #             # TODO: Debugging, here is a type instability
#     #             global_res = MPI.Reduce!(Ref(res), max, mpi_root(), mpi_comm())
#     #             if mpi_isroot()
#     #                 res::eltype(du) = global_res[]
#     #             end
#     #         end
#     #         if mpi_isroot()
#     #             @printf("  % 10.8e", res)
#     #             @printf(io, "  % 10.8e", res)
#     #         end
#     #     end
#     #     println()
#     # end

#     # # L2/L∞ errors of the primitive variables
#     # if :l2_error_primitive in analysis_errors ||
#     #    :linf_error_primitive in analysis_errors
#     #     l2_error_prim, linf_error_prim = calc_error_norms(cons2prim, u_ode, t, analyzer,
#     #                                                       semi, cache_analysis)

#     #     if mpi_isroot()
#     #         print(" Variable:    ")
#     #         for v in eachvariable(equations)
#     #             @printf("   %-14s", varnames(cons2prim, equations)[v])
#     #         end
#     #         println()

#     #         # L2 error
#     #         if :l2_error_primitive in analysis_errors
#     #             print(" L2 error prim.: ")
#     #             for v in eachvariable(equations)
#     #                 @printf("%10.8e   ", l2_error_prim[v])
#     #                 @printf(io, "  % 10.8e", l2_error_prim[v])
#     #             end
#     #             println()
#     #         end

#     #         # L∞ error
#     #         if :linf_error_primitive in analysis_errors
#     #             print(" Linf error pri.:")
#     #             for v in eachvariable(equations)
#     #                 @printf("%10.8e   ", linf_error_prim[v])
#     #                 @printf(io, "  % 10.8e", linf_error_prim[v])
#     #             end
#     #             println()
#     #         end
#     #     end
#     # end

#     # # additional integrals
#     # analyze_integrals(analysis_integrals, io, du, u, t, semi)

#     return nothing
# end

# function entropy_timederivative end
# pretty_form_utf(::typeof(entropy_timederivative)) = "∑∂S/∂U ⋅ Uₜ"
# pretty_form_ascii(::typeof(entropy_timederivative)) = "dsdu_ut"

# pretty_form_utf(::typeof(entropy)) = "∑S"

# pretty_form_utf(::typeof(energy_total)) = "∑e_total"
# pretty_form_ascii(::typeof(energy_total)) = "e_total"

# pretty_form_utf(::typeof(energy_kinetic)) = "∑e_kinetic"
# pretty_form_ascii(::typeof(energy_kinetic)) = "e_kinetic"

# pretty_form_utf(::typeof(energy_kinetic_nondimensional)) = "∑e_kinetic*"
# pretty_form_ascii(::typeof(energy_kinetic_nondimensional)) = "e_kinetic*"

# pretty_form_utf(::typeof(energy_internal)) = "∑e_internal"
# pretty_form_ascii(::typeof(energy_internal)) = "e_internal"

# pretty_form_utf(::typeof(energy_magnetic)) = "∑e_magnetic"
# pretty_form_ascii(::typeof(energy_magnetic)) = "e_magnetic"

# pretty_form_utf(::typeof(cross_helicity)) = "∑v⋅B"
# pretty_form_ascii(::typeof(cross_helicity)) = "v_dot_B"

# pretty_form_utf(::typeof(enstrophy)) = "∑enstrophy"
# pretty_form_ascii(::typeof(enstrophy)) = "enstrophy"

# pretty_form_utf(::Val{:l2_divb}) = "L2 ∇⋅B"
# pretty_form_ascii(::Val{:l2_divb}) = "l2_divb"

# pretty_form_utf(::Val{:linf_divb}) = "L∞ ∇⋅B"
# pretty_form_ascii(::Val{:linf_divb}) = "linf_divb"

# pretty_form_utf(::typeof(lake_at_rest_error)) = "∑|H₀-(h+b)|"
# pretty_form_ascii(::typeof(lake_at_rest_error)) = "|H0-(h+b)|"

# specialized implementations specific to some solvers
# PointCloudDomain (move later)
# get_component(u::StructArray, i::Int) = StructArrays.component(u, i)
# get_component(u::AbstractArray{<:SVector}, i::Int) = getindex.(u, i)
function create_cache_analysis(analyzer, mesh::PointCloudDomain,
                               equations, solver::PointCloudSolver, cache,
                               RealT, uEltype)
    # md = mesh.md
    return (;)
end

SolutionAnalyzer(pd::RefPointData) = pd

nelements(mesh::PointCloudDomain, ::PointCloudSolver, other_args...) = mesh.pd.num_points
function ndofsglobal(mesh::PointCloudDomain, solver::PointCloudSolver, cache)
    return Trixi.ndofs(mesh, solver, cache)
end
end # @muladd