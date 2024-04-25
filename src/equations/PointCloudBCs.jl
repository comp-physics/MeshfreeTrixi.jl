# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Use to strongly impose BCs and zero du
struct FluxZero end

function (surface_flux::FluxZero)(u_inner, u_boundary, normal_direction, equations)
    SVector(zeros(length(u_inner))...)
end

@inline function apply_slip_velocity(u, normal_vector,
                                     equations::CompressibleEulerEquations2D)
    v = SVector(u[2], u[3])
    # m_bound .= m_bound .- (m_bound ⋅ boundary_normals[5][i]) * boundary_normals[5][i]
    v_slip = v .- (v ⋅ normal_vector) * normal_vector
    return SVector(u[1], v_slip[1], v_slip[2], u[4])
end

"""
    BoundaryConditionDirichlet(boundary_value_function)

Create a Dirichlet boundary condition that uses the function `boundary_value_function`
to specify the values at the boundary.
This can be used to create a boundary condition that specifies exact boundary values
by passing the exact solution of the equation.
The passed boundary value function will be called with the same arguments as an initial condition function is called, i.e., as
```julia
boundary_value_function(x, t, equations)
```
where `x` specifies the coordinates, `t` is the current time, and `equation` is the corresponding system of equations.

# Examples
```julia
julia> BoundaryConditionDirichlet(initial_condition_convergence_test)
```
"""
# struct BoundaryConditionDirichlet{B}
#     boundary_value_function::B
# end

# Dirichlet-type boundary condition for use with UnstructuredMesh2D
# Note: For unstructured we lose the concept of an "absolute direction"
# Modified for Point Cloud implementation
# requires strongly imposing boundary conditions
@inline function (boundary_condition::BoundaryConditionDirichlet)(du_inner, u_inner,
                                                                  normal_direction::AbstractVector,
                                                                  x, t,
                                                                  surface_flux_function::FluxZero,
                                                                  equations)
    # get the external value of the solution
    u_boundary = boundary_condition.boundary_value_function(x, t, equations)
    # u_inner = u_boundary

    # Calculate boundary flux
    # Will always return zero vector
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux, u_boundary
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::CompressibleEulerEquations2D)

Determine the boundary numerical surface flux for a slip wall condition.
Imposes a zero normal velocity at the wall.
Density is taken from the internal solution state and pressure is computed as an
exact solution of a 1D Riemann problem. Further details about this boundary state
are available in the paper:
- J. J. W. van der Vegt and H. van der Ven (2002)
  Slip flow boundary conditions in discontinuous Galerkin discretizations of
  the Euler equations of gas dynamics
  [PDF](https://reports.nlr.nl/bitstream/handle/10921/692/TP-2002-300.pdf?sequence=1)

Details about the 1D pressure Riemann solution can be found in Section 6.3.3 of the book
- Eleuterio F. Toro (2009)
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
  3rd edition
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

Should be used together with [`UnstructuredMesh2D`](@ref).
"""
@inline function boundary_condition_slip_wall(du_inner, u_inner,
                                              normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function::FluxZero,
                                              equations::CompressibleEulerEquations2D)
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # rotate the internal solution state
    u_local = apply_slip_velocity(u_inner, normal, equations)

    # For the slip wall we directly set the flux as the normal velocity is zero
    # Strongly imposed, hardset du to 0
    return SVector(du_inner[1],
                   zero(eltype(u_inner)),
                   zero(eltype(u_inner)),
                   du_inner[4]),
           u_local
end

struct BoundaryConditionDoNothing end

@inline function (::BoundaryConditionDoNothing)(du_inner, u_inner,
                                                outward_direction::AbstractVector,
                                                x, t, surface_flux::FluxZero, equations)
    return du_inner,
           u_inner
end
end # @muladd
