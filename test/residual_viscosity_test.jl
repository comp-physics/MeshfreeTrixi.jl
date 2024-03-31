using MeshfreeTrixi
using RadialBasisFiniteDifferences
using LinearAlgebra
using SparseArrays

# Specialized Methods
approximation_order = 3
basis = PointCloudBasis(Point2D(), approximation_order;
                        approximation_type = RBF(PolyharmonicSpline(5)))
solver = PointCloudSolver(basis)

casename = "data/cyl_0_05"
boundary_names = Dict(:inlet => 1, :outlet => 2, :bottom => 3, :top => 4, :cyl => 5)
domain = PointCloudDomain(solver, casename, boundary_names)

# Instantiate Semidiscretization
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
equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_gradient
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => BoundaryConditionNeumann(initial_condition),
                       :rest => boundary_condition_slip_wall)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

u0 = ode.u0
u1 = deepcopy(u0)
du0 = deepcopy(u0)
du1 = deepcopy(u0)

Dx0 = semi.cache.rbf_differentiation_matrices[1]
Dy0 = semi.cache.rbf_differentiation_matrices[2]

Trixi.apply_to_each_field(Trixi.mul_by!(Dx0), du0, u0)

rbfdeg = 5; # PHS power (r^p).
polydeg = approximation_order; # Augmented polynomial degree.
n = domain.pd.num_neighbors# Stencil size.

### Generate Global Operator Matrices from Local RBF Operator Matrices
E1, Dx1, Dy1, Dxx1, Dyy1, Dxy1 = generate_operator(domain.pd.points, domain.pd.points,
                                                   rbfdeg, n, polydeg)

mul!(du1, Dx1, u1)

### Generate Hyperviscosity Operator
# k = 1 # 2nd Derivative
# Dxk, Dyk = hyperviscosity_operator(2 * k, X, X, rbfdeg, n, polydeg)

# These fail due to conditioning
# @assert Dx0 ≈ Dx1
# @assert Dy0 ≈ Dy1

Trixi.apply_to_each_field(Trixi.mul_by!(Dx0), du0, u0)
Trixi.apply_to_each_field(Trixi.mul_by!(Dx1), du1, u1)
@test du0 ≈ du1

Trixi.apply_to_each_field(Trixi.mul_by!(Dy0), du0, u0)
Trixi.apply_to_each_field(Trixi.mul_by!(Dy1), du1, u1)
@test du0 ≈ du1

# Test application of operator

# # Test source_terms
# source_hv = SourceHyperviscosityFlyer(solver, equations, domain; k = 2, c = 1.0)
# source_hv2 = SourceHyperviscosityTominec(solver, equations, domain; c = 1.0)
# sources = (; source_hv, source_hv2)
# sources = SourceTerms(hv = source_hv, hv2 = source_hv2)
# semi = SemidiscretizationHyperbolic(domain, equations,
#                                     initial_condition, solver;
#                                     boundary_conditions = boundary_conditions,
#                                     source_terms = sources)
# tspan = (0.0, 0.4)
# ode = semidiscretize(semi, tspan)

# # Test history callback
# history_callback = HistoryCallback(approx_order = approximation_order)
# source_rv = SourceResidualViscosityTominec(solver, equations, domain; c = 1.0,
#                                            polydeg = approximation_order)
# sources = SourceTerms(hv = source_hv2, rv = source_rv)
# semi = SemidiscretizationHyperbolic(domain, equations,
#                                     initial_condition, solver;
#                                     boundary_conditions = boundary_conditions,
#                                     source_terms = sources)
# tspan = (0.0, 0.4)
# ode = semidiscretize(semi, tspan)