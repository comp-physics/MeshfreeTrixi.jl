# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    VTKSaveSolutionCallback(; interval::Integer=0,
                           dt=nothing,
                           save_initial_solution=true,
                           save_final_solution=true,
                           output_directory="out",
                           solution_variables=cons2prim)

Save the current numerical solution in regular intervals. Either pass `interval` to save
every `interval` time steps or pass `dt` to save in intervals of `dt` in terms
of integration time by adding additional (shortened) time steps where necessary (note that this may change the solution).
`solution_variables` can be any callable that converts the conservative variables
at a single point to a set of solution variables. The first parameter passed
to `solution_variables` will be the set of conservative variables
and the second parameter is the equation struct.
"""
mutable struct VTKSaveSolutionCallback{IntervalType, SolutionVariablesType}
    interval_or_dt::IntervalType
    save_initial_solution::Bool
    save_final_solution::Bool
    output_directory::String
    solution_variables::SolutionVariablesType
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:VTKSaveSolutionCallback})
    @nospecialize cb # reduce precompilation time

    save_solution_callback = cb.affect!
    print(io, "VTKSaveSolutionCallback(interval=",
          save_solution_callback.interval_or_dt,
          ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:VTKSaveSolutionCallback}})
    @nospecialize cb # reduce precompilation time

    save_solution_callback = cb.affect!.affect!
    print(io, "VTKSaveSolutionCallback(dt=", save_solution_callback.interval_or_dt, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:VTKSaveSolutionCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        save_solution_callback = cb.affect!

        setup = [
            "interval" => save_solution_callback.interval_or_dt,
            "solution variables" => save_solution_callback.solution_variables,
            "save initial solution" => save_solution_callback.save_initial_solution ?
                                       "yes" : "no",
            "save final solution" => save_solution_callback.save_final_solution ?
                                     "yes" : "no",
            "output directory" => abspath(normpath(save_solution_callback.output_directory))
        ]
        summary_box(io, "VTKSaveSolutionCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:VTKSaveSolutionCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        save_solution_callback = cb.affect!.affect!

        setup = [
            "dt" => save_solution_callback.interval_or_dt,
            "solution variables" => save_solution_callback.solution_variables,
            "save initial solution" => save_solution_callback.save_initial_solution ?
                                       "yes" : "no",
            "save final solution" => save_solution_callback.save_final_solution ?
                                     "yes" : "no",
            "output directory" => abspath(normpath(save_solution_callback.output_directory))
        ]
        summary_box(io, "VTKSaveSolutionCallback", setup)
    end
end

function VTKSaveSolutionCallback(; interval::Integer = 0,
                                 dt = nothing,
                                 save_initial_solution = true,
                                 save_final_solution = true,
                                 output_directory = "out",
                                 solution_variables = cons2prim)
    if !isnothing(dt) && interval > 0
        throw(ArgumentError("You can either set the number of steps between output (using `interval`) or the time between outputs (using `dt`) but not both simultaneously"))
    end

    # Expected most frequent behavior comes first
    if isnothing(dt)
        interval_or_dt = interval
    else # !isnothing(dt)
        interval_or_dt = dt
    end

    solution_callback = VTKSaveSolutionCallback(interval_or_dt,
                                                save_initial_solution,
                                                save_final_solution,
                                                output_directory, solution_variables)

    # Expected most frequent behavior comes first
    if isnothing(dt)
        # Save every `interval` (accepted) time steps
        # The first one is the condition, the second the affect!
        return DiscreteCallback(solution_callback, solution_callback,
                                save_positions = (false, false),
                                initialize = initialize_save_cb!)
    else
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(solution_callback, dt,
                                save_positions = (false, false),
                                initialize = initialize_save_cb!,
                                final_affect = save_final_solution)
    end
end

function initialize_save_cb!(cb, u, t, integrator)
    # The VTKSaveSolutionCallback is either cb.affect! (with DiscreteCallback)
    # or cb.affect!.affect! (with PeriodicCallback).
    # Let recursive dispatch handle this.
    initialize_save_cb!(cb.affect!, u, t, integrator)
end

function initialize_save_cb!(solution_callback::VTKSaveSolutionCallback, u, t,
                             integrator)
    mpi_isroot() && mkpath(solution_callback.output_directory)

    semi = integrator.p
    # @trixi_timeit timer() "I/O" save_mesh(semi, solution_callback.output_directory)

    if solution_callback.save_initial_solution
        solution_callback(integrator)
    end

    return nothing
end

# # Save mesh for a general semidiscretization (default)
# function save_mesh(semi::AbstractSemidiscretization, output_directory, timestep = 0)
#     mesh, _, _, _ = mesh_equations_solver_cache(semi)

#     if mesh.unsaved_changes
#         # We only append the time step number to the mesh file name if it has
#         # changed during the simulation due to AMR. We do not append it for
#         # the first time step.
#         if timestep == 0
#             mesh.current_filename = save_mesh_file(mesh, output_directory)
#         else
#             mesh.current_filename = save_mesh_file(mesh, output_directory, timestep)
#         end
#         mesh.unsaved_changes = false
#     end
# end

# this method is called to determine whether the callback should be activated
function (solution_callback::VTKSaveSolutionCallback)(u, t, integrator)
    @unpack interval_or_dt, save_final_solution = solution_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval_or_dt > 0 && (((integrator.stats.naccept % interval_or_dt == 0) &&
             !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
            (save_final_solution && isfinished(integrator)))
end

# this method is called when the callback is activated
function (solution_callback::VTKSaveSolutionCallback)(integrator)
    u_ode = integrator.u
    semi = integrator.p
    iter = integrator.stats.naccept

    @trixi_timeit timer() "I/O" begin
        # Call high-level functions that dispatch on semidiscretization type
        # @trixi_timeit timer() "save mesh" save_mesh(semi,
        #                                             solution_callback.output_directory,
        #                                             iter)
        save_solution_file_vtk(semi, u_ode, solution_callback, integrator)
    end

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

@inline function save_solution_file_vtk(semi::AbstractSemidiscretization, u_ode,
                                        solution_callback,
                                        integrator; system = "")
    @unpack t, dt = integrator
    iter = integrator.stats.naccept

    element_variables = Dict{Symbol, Any}()
    @trixi_timeit timer() "get element variables" begin
        get_element_variables!(element_variables, u_ode, semi)
        callbacks = integrator.opts.callback
        if callbacks isa CallbackSet
            foreach(callbacks.continuous_callbacks) do cb
                get_element_variables!(element_variables, u_ode, semi, cb;
                                       t = integrator.t, iter = iter)
            end
            foreach(callbacks.discrete_callbacks) do cb
                get_element_variables!(element_variables, u_ode, semi, cb;
                                       t = integrator.t, iter = iter)
            end
        end
    end

    node_variables = Dict{Symbol, Any}()
    @trixi_timeit timer() "get node variables" get_node_variables!(node_variables,
                                                                   semi)

    @trixi_timeit timer() "save solution" save_solution_file_vtk(u_ode, t, dt, iter,
                                                                 semi,
                                                                 solution_callback,
                                                                 element_variables,
                                                                 node_variables,
                                                                 system = system)
end

@inline function save_solution_file_vtk(u_ode, t, dt, iter,
                                        semi::AbstractSemidiscretization,
                                        solution_callback,
                                        element_variables = Dict{Symbol, Any}(),
                                        node_variables = Dict{Symbol, Any}();
                                        system = "")
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array_native(u_ode, mesh, equations, solver, cache)
    save_solution_file_vtk(u, t, dt, iter, mesh, equations, solver, cache,
                           solution_callback,
                           element_variables,
                           node_variables; system = system)
end

# TODO: Taal refactor, move save_mesh_file?
# function save_mesh_file(mesh::TreeMesh, output_directory, timestep=-1) in io/io.jl

function save_solution_file_vtk(u, time, dt, timestep,
                                mesh::Union{SerialTreeMesh, StructuredMesh,
                                            UnstructuredMesh2D, SerialP4estMesh,
                                            SerialT8codeMesh},
                                equations, dg::DG, cache,
                                solution_callback,
                                element_variables = Dict{Symbol, Any}(),
                                node_variables = Dict{Symbol, Any}();
                                system = "")
    @unpack output_directory, solution_variables = solution_callback

    # Filename based on current time step
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%06d.h5", timestep))
    else
        filename = joinpath(output_directory,
                            @sprintf("solution_%s_%06d.h5", system, timestep))
    end

    # Convert to different set of variables if requested
    if solution_variables === cons2cons
        data = u
        n_vars = nvariables(equations)
    else
        # Reinterpret the solution array as an array of conservative variables,
        # compute the solution variables via broadcasting, and reinterpret the
        # result as a plain array of floating point numbers
        data = Array(reinterpret(eltype(u),
                                 solution_variables.(reinterpret(SVector{nvariables(equations),
                                                                         eltype(u)}, u),
                                                     Ref(equations))))

        # Find out variable count by looking at output from `solution_variables` function
        n_vars = size(data, 1)
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nelements(dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data
        for v in 1:n_vars
            # Convert to 1D array
            file["variables_$v"] = vec(data[v, .., :])

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = varnames(solution_variables, equations)[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file
            file["element_variables_$v"] = element_variable

            # Add variable name as attribute
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end

        # Store node variables
        for (v, (key, node_variable)) in enumerate(node_variables)
            # Add to file
            file["node_variables_$v"] = node_variable

            # Add variable name as attribute
            var = file["node_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end
end # @muladd
