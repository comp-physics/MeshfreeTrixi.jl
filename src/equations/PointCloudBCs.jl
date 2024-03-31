# Use to strongly impose BCs and zero du
struct FluxZero end

function (surface_flux::FluxZero)(u_inner, u_boundary, normal_direction, equations)
  SVector(zeros(length(u_inner))...)
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
@inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner,
  normal_direction::AbstractVector,
  x, t,
  surface_flux_function::FluxZero,
  equations)
  # get the external value of the solution
  u_boundary = boundary_condition.boundary_value_function(x, t, equations)
  u_inner = u_boundary

  # Calculate boundary flux
  # Will always return zero vector
  flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

  return flux
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
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
  x, t,
  surface_flux_function::FluxZero,
  equations::CompressibleEulerEquations2D)
  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal = normal_direction / norm_

  # rotate the internal solution state
  u_local = Trixi.rotate_to_x(u_inner, normal, equations)
  u_inner = u_local

  # compute the primitive variables
  # rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

  # # Get the solution of the pressure Riemann problem
  # # See Section 6.3.3 of
  # # Eleuterio F. Toro (2009)
  # # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
  # # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
  # if v_normal <= 0.0
  #     sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
  #     p_star = p_local *
  #              (1 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2 *
  #                                                                          equations.gamma *
  #                                                                          equations.inv_gamma_minus_one)
  # else # v_normal > 0.0
  #     A = 2 / ((equations.gamma + 1) * rho_local)
  #     B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
  #     p_star = p_local +
  #              0.5 * v_normal / A *
  #              (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
  # end

  # For the slip wall we directly set the flux as the normal velocity is zero
  # Strongly imposed, hardset du to 0
  return SVector(zero(eltype(u_inner)),
    zero(eltype(u_inner)),
    zero(eltype(u_inner)),
    zero(eltype(u_inner)))
end