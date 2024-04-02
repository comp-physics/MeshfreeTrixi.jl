using MeshfreeTrixi
using OrdinaryDiffEq
using SimpleUnPack

# includet("../header.jl")

# Base Methods
approximation_order = 3
rbf_order = 3
# Specialized Methods
basis = PointCloudBasis(Point2D(), approximation_order;
                        approximation_type = RBF(PolyharmonicSpline(rbf_order)))
solver = PointCloudSolver(basis)

casename = "data/cyl_0_05"
boundary_names = Dict(:inlet => 1, :outlet => 2, :bottom => 3, :top => 4, :cyl => 5)
domain = PointCloudDomain(solver, casename, boundary_names)

# Instantiate Semidiscretization
equations = CompressibleEulerEquations2D(1.4)
function initial_condition_cyl(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.4
    rho_v1 = 4.1
    rho_v2 = 0.0
    rho_e = 8.8 #* 1.4
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function initial_condition_gradient(x, t, equations::CompressibleEulerEquations2D)
    rho_s = 1.4
    rho_v1_s = 4.1
    rho_v2_s = 0.0
    rho_e_s = 8.8
    rho = rho_s + 0.1 * x[1] + 0.1 * x[2]
    rho_v1 = rho_v1_s + 0.1 * x[1] + 0.1 * x[2]
    rho_v2 = rho_v2_s + 0.1 * x[1] + 0.1 * x[2]
    rho_e = rho_e_s + 0.1 * x[1] + 0.1 * x[2]
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
initial_condition = initial_condition_gradient
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => BoundaryConditionDoNothing(),
                       :top => boundary_condition_slip_wall,
                       :bottom => boundary_condition_slip_wall,
                       :cyl => boundary_condition_slip_wall)

# Test upwind viscosity
source_rv = SourceUpwindViscosityTominec(solver, equations, domain)
source_hv2 = SourceHyperviscosityTominec(solver, equations, domain; c = 1.0)
sources = SourceTerms(hv = source_hv2, rv = source_rv)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

# Try inner operators
u0 = ode.u0
u1 = deepcopy(u0)
du0 = deepcopy(u0)
du1 = deepcopy(u0)

# Function
@unpack mesh, solver, cache, equations = semi
MeshfreeTrixi.set_to_zero!(du0)
MeshfreeTrixi.calc_fluxes!(du0, u0, domain,
                           Trixi.have_nonconservative_terms(equations), equations,
                           solver.engine, solver, cache)
# Manual
Dx = semi.cache.rbf_differentiation_matrices[1]
Dy = semi.cache.rbf_differentiation_matrices[2]
flux_values_x = deepcopy(u0)
flux_values_y = deepcopy(u0)
for e in eachelement(domain, solver, cache)
    flux_values_x[e] = flux(u1[e], 1, equations)
end
for e in eachelement(domain, solver, cache)
    flux_values_y[e] = flux(u1[e], 2, equations)
end
du1 = -Dx * flux_values_x - Dy * flux_values_y

@test du0 ≈ du1
