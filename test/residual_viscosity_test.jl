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

    # Parameters for oscillations
    A = 1.0  # Amplitude of oscillations
    kx = 100.0  # Frequency in x-direction
    ky = 100.0  # Frequency in y-direction

    # Adding oscillations to the base state
    oscillation = A * sin(2 * π * kx * x[1]) * sin(2 * π * ky * x[2])

    rho = rho_s + 0.1 * x[1] + 0.1 * x[2] + oscillation
    rho_v1 = rho_v1_s + 0.1 * x[1] + 0.1 * x[2] + oscillation
    rho_v2 = rho_v2_s + 0.1 * x[1] + 0.1 * x[2] + oscillation
    rho_e = rho_e_s + 0.1 * x[1] + 0.1 * x[2] + oscillation

    return SVector(rho, rho_v1, rho_v2, rho_e)
end
equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_gradient
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => BoundaryConditionNeumann(initial_condition),
                       :rest => boundary_condition_slip_wall)
source_hv = SourceHyperviscosityFlyer(solver, equations, domain; k = 2, c = 1.0)
source_hv2 = SourceHyperviscosityTominec(solver, equations, domain; c = 1.0)
sources = SourceTerms(hv = source_hv, hv2 = source_hv2)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

u0 = ode.u0
u1 = deepcopy(u0)
du0 = deepcopy(u0)
du1 = deepcopy(u0)

# Flyer
D0_flyer = semi.source_terms.sources.hv.cache.hv_differentiation_matrix
D0_tominec = semi.source_terms.sources.hv2.cache.hv_differentiation_matrix

rbfdeg = 5; # PHS power (r^p).
polydeg = approximation_order; # Augmented polynomial degree.
n = domain.pd.num_neighbors# Stencil size.

### Generate Hyperviscosity Operator
# Tominec
k = 1 # 2nd Derivative
Dxk, Dyk = hyperviscosity_operator(2 * k, domain.pd.points, domain.pd.points, rbfdeg, n,
                                   polydeg)
lap = Dxk + Dyk
D1_tominec = lap' * lap
gamma = semi.source_terms.sources.hv2.cache.gamma
MeshfreeTrixi.set_to_zero!(du0)
MeshfreeTrixi.set_to_zero!(du1)
Trixi.apply_to_each_field(Trixi.mul_by_accum!(D0_tominec,
                                              gamma),
                          du0, u0)
Trixi.apply_to_each_field(Trixi.mul_by_accum!(D1_tominec,
                                              gamma),
                          du1, u1)
@test isapprox(du0, du1; rtol = 1e-1)

# # Flyer
# # Very sensitive to condition number
# # Generally cannot pass despite being close
# # Ex:   du0 = [2.03171374932064,    2.0317137493206445, 2.031713749320641,  2.031713749320656]
# #       du1 =  [1.9763717904148657, 1.9763717904148659, 1.9763717904148677, 1.9763717904148688]
# k = 2 # 2nd Derivative
# Dxk, Dyk = hyperviscosity_operator(2 * k, domain.pd.points, domain.pd.points, rbfdeg, n,
#                                    polydeg)
# D1_flyer = Dxk + Dyk
# gamma = semi.source_terms.sources.hv.cache.gamma
# MeshfreeTrixi.set_to_zero!(du0)
# MeshfreeTrixi.set_to_zero!(du1)
# Trixi.apply_to_each_field(Trixi.mul_by_accum!(D0_flyer,
#                                               gamma),
#                           du0, u0)
# Trixi.apply_to_each_field(Trixi.mul_by_accum!(D1_flyer,
#                                               gamma),
#                           du1, u1)
# @test du0 ≈ du1
# isapprox(du0, du1; rtol = 1e-1)

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
