# Code based on TrixiParticles.jl
# Helper functions 
function assemble_systems(semi)
    # Check if `source_terms.sources` exists
    if hasproperty(semi.source_terms, :sources)
        # `source_terms.sources` exists, use it
        return (semi.equations, semi.source_terms.sources...)
    else
        # `source_terms.sources` does not exist, fall back to `source_terms`
        return (semi.equations, semi.source_terms)
    end
end
# Same as `foreach`, but it optimizes away for small input tuples
@inline function foreach_noalloc(func, collection)
    element = first(collection)
    remaining_collection = Base.tail(collection)

    func(element)

    # Process remaining collection
    foreach_noalloc(func, remaining_collection)
end

@inline foreach_noalloc(func, collection::Tuple{}) = nothing
# This is just for readability to loop over all systems without allocations
# We construct our own "system" instead of containing on in semi
# @inline foreach_system(f, semi::Union{NamedTuple, Semidiscretization}) = foreach_noalloc(f,
#                                                                                          semi.systems)
@inline foreach_system(f, systems) = foreach_noalloc(f, systems)
@inline function system_indices(system, systems)
    # Note that this takes only about 5 ns, while mapping systems to indices with a `Dict`
    # is ~30x slower because `hash(::System)` is very slow.
    index = findfirst(==(system), systems)

    if isnothing(index)
        throw(ArgumentError("system is not in the semidiscretization"))
    end

    return index
end
# File system methods
vtkname(system) = string(nameof(typeof(system)))
# vtkname(system::SolidSystem) = "solid"
# vtkname(system::BoundarySystem) = "boundary"
function system_names(systems)
    # Add `_i` to each system name, where `i` is the index of the corresponding
    # system type.
    # `["fluid", "boundary", "boundary"]` becomes `["fluid_1", "boundary_1", "boundary_2"]`.
    cnames = systems .|> vtkname
    filenames = [string(cnames[i], "_", count(==(cnames[i]), cnames[1:i]))
                 for i in eachindex(cnames)]
    return filenames
end
function get_git_hash()
    pkg_directory = pkgdir(@__MODULE__)
    git_directory = joinpath(pkg_directory, ".git")

    # Check if the .git directory exists
    if !isdir(git_directory)
        return "UnknownVersion"
    end

    try
        git_cmd = Cmd(`git describe --tags --always --first-parent --dirty`,
                      dir = pkg_directory)
        return string(readchomp(git_cmd))
    catch e
        return "UnknownVersion"
    end
end

### TODO: Have all quantities save to one vtk file
"""
    trixi2vtk(u_ode, semi, t; iter=nothing, output_directory="out", prefix="",
              write_meta_data=true, max_coordinates=Inf, custom_quantities...)

Convert MF-Trixi simulation data to VTK format.

# Arguments
- `u_ode`: Solution of the MF-Trixi ODE system at one time step.
            This expects an `ArrayPartition` as returned in the examples as `sol.u[end]`.
- `semi`:   Semidiscretization of the MF-Trixi simulation.
- `t`:      Current time of the simulation.

# Keywords
- `iter=nothing`:           Iteration number when multiple iterations are to be stored in
                            separate files. This number is just appended to the filename.
- `output_directory="out"`: Output directory path.
- `prefix=""`:              Prefix for output files.
- `write_meta_data=true`:   Write meta data.
- `max_coordinates=Inf`     The coordinates of particles will be clipped if their absolute
                            values exceed this threshold.
- `custom_quantities...`:   Additional custom quantities to include in the VTK output.
                            Each custom quantity must be a function of `(u, u, t, system)`,
                            which will be called for every system, where `u` and `u` are the
                            wrapped solution arrays for the corresponding system and `t` is
                            the current simulation time. Note that working with these `u`
                            and `u` arrays requires undocumented internal functions of
                            MF-Trixi. See [Custom Quantities](@ref custom_quantities)
                            for a list of pre-defined custom quantities that can be used here.

# Example
```jldoctest; output = false, setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing))
trixi2vtk(sol.u[end], semi, 0.0, iter=1, output_directory="output", prefix="solution")

# Additionally store the kinetic energy of each system as "my_custom_quantity"
trixi2vtk(sol.u[end], semi, 0.0, iter=1, my_custom_quantity=kinetic_energy)

# output

```
"""
function trixi2vtk(u_ode, semi, t; iter = nothing, output_directory = "out", prefix = "",
                   write_meta_data = true, max_coordinates = Inf, custom_quantities...)
    # Package together objects to be saved 
    systems = assemble_systems(semi)

    filenames = system_names(systems)

    foreach_system(systems) do system
        system_index = system_indices(system, systems)

        u = Trixi.wrap_array(u_ode, semi.mesh, semi.equations, semi.solver, semi.cache)

        trixi2vtk(u, t, system, semi;
                  output_directory = output_directory,
                  system_name = filenames[system_index], iter = iter, prefix = prefix,
                  write_meta_data = write_meta_data, max_coordinates = max_coordinates,
                  custom_quantities...)
    end
end

# Convert data for a single TrixiParticle system to VTK format
function trixi2vtk(u, t, system, semi; output_directory = "out", prefix = "",
                   iter = nothing, system_name = vtkname(system), write_meta_data = true,
                   max_coordinates = Inf,
                   custom_quantities...)
    mkpath(output_directory)

    # handle "_" on optional pre/postfix strings
    add_opt_str_pre(str) = (str === "" ? "" : "$(str)_")
    add_opt_str_post(str) = (str === nothing ? "" : "_$(str)")

    file = joinpath(output_directory,
                    add_opt_str_pre(prefix) * "$system_name"
                    * add_opt_str_post(iter))

    collection_file = joinpath(output_directory,
                               add_opt_str_pre(prefix) * "$system_name")

    # Reset the collection when the iteration is 0
    pvd = paraview_collection(collection_file; append = iter > 0)

    # points = periodic_coords(current_coordinates(u, system))
    points = reduce(hcat, semi.mesh.pd.points) # TODO: Remove allocations here
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    if abs(maximum(points)) > max_coordinates || abs(minimum(points)) > max_coordinates
        println("Warning: At least one particle's absolute coordinates exceed `max_coordinates`"
                *
                " and have been clipped")
        for i in eachindex(points)
            points[i] = clamp(points[i], -max_coordinates, max_coordinates)
        end
    end

    save_tag = string(nameof(typeof(system)))
    vtk_grid(file, points, cells) do vtk
        @trixi_timeit timer() "save $save_tag" write2vtk!(vtk, u, t, system, semi,
                                                          write_meta_data = write_meta_data)

        # Store particle index
        vtk["index"] = eachelement(semi.mesh, semi.solver, semi.cache)
        # vtk["index"] = eachparticle(system)
        vtk["time"] = t

        if write_meta_data
            vtk["solver_version"] = get_git_hash()
            vtk["julia_version"] = string(VERSION)
        end

        # Extract custom quantities for this system
        for (key, quantity) in custom_quantities
            value = custom_quantity(quantity, u, t, system)
            if value !== nothing
                vtk[string(key)] = value
            end
        end

        # Add to collection
        pvd[t] = vtk
    end
    vtk_save(pvd)
end

function custom_quantity(quantity::AbstractArray, u, t, system)
    return quantity
end

function custom_quantity(quantity, u, t, system)
    # Assume `quantity` is a function of `u`, `u`, `t`, and `system`
    return quantity(u, u, t, system)
end

"""
    trixi2vtk(coordinates; output_directory="out", prefix="", filename="coordinates")

Convert coordinate data to VTK format.

# Arguments
- `coordinates`: Coordinates to be saved.

# Keywords
- `output_directory="out"`: Output directory path.
- `prefix=""`:              Prefix for the output file.
- `filename="coordinates"`: Name of the output file.

# Returns
- `file::AbstractString`: Path to the generated VTK file.
"""
function trixi2vtk(coordinates; output_directory = "out", prefix = "",
                   filename = "coordinates",
                   custom_quantities...)
    mkpath(output_directory)
    file = prefix === "" ? joinpath(output_directory, filename) :
           joinpath(output_directory, "$(prefix)_$filename")

    points = coordinates
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(file, points, cells) do vtk
        # Store particle index.
        vtk["index"] = [i for i in axes(coordinates, 2)]

        # Extract custom quantities for this system.
        for (key, quantity) in custom_quantities
            if quantity !== nothing
                vtk[string(key)] = quantity
            end
        end
    end

    return file
end

### Instead of system::SysType, use a combination of Eqns and Source Types to dispatch
function write2vtk!(vtk, u, t, system, semi; write_meta_data)
    return vtk
end
function write2vtk!(vtk, u, t, system::CompressibleEulerEquations2D, semi;
                    write_meta_data = true)
    # Export conservative variables
    vtk["density"] = get_component(u, 1)
    vtk["density_energy"] = get_component(u, 4)
    vtk["momentum"] = vcat(get_component(u, 2)', get_component(u, 3)')

    # Post-process primitive variables
    u_prim = semi.cache.local_values_threaded[1]
    for i in eachindex(u_prim)
        u_prim[i] = cons2prim(u[i], system)
    end
    vtk["pressure"] = get_component(u_prim, 4)
    vtk["velocity"] = vcat(get_component(u_prim, 2)', get_component(u_prim, 3)')

    return vtk
end

function write2vtk!(vtk, u, t, system::SourceUpwindViscosityTominec, semi;
                    write_meta_data = true)
    vtk["eps"] = system.cache.eps
    vtk["eps_scalar"] = system.cache.eps_c

    return vtk
end

function write2vtk!(vtk, u, t, system::SourceResidualViscosityTominec, semi;
                    write_meta_data = true)
    vtk["eps"] = system.cache.eps
    vtk["eps_scalar"] = system.cache.eps_c
    vtk["eps_uw"] = system.cache.eps_uw
    vtk["eps_rv"] = system.cache.eps_rv

    return vtk
end

function write2vtk!(vtk, u, t, system::SourceIGR, semi;
                    write_meta_data = true)
    vtk["sigma"] = system.cache.sigma

    return vtk
end

write2vtk!(vtk, viscosity::Nothing) = vtk
