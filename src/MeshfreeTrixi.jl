"""
    MeshfreeTrixi

**MeshfreeTrixi.jl** is an extension of the Trixi numerical simulation framework 
for hyperbolic conservation laws. This extension implements a variety of meshfree
numerical simulation methods primarily based on radial basis functions (RBFs).
We leverage the code structure of Trixi.jl to provide a consistent interface for
the user and to allow for easy integration with DG solvers and methods.
"""
module MeshfreeTrixi
# Include other packages that are used in MeshfreeTrixi.jl
# (standard library packages first, other packages next, all of them sorted alphabetically)

using LinearAlgebra: LinearAlgebra, Diagonal, diag, dot, mul!, norm, cross, normalize, I,
                     UniformScaling, det
using Printf: @printf, @sprintf, println
using SparseArrays: AbstractSparseMatrix, AbstractSparseMatrixCSC, sparse, droptol!,
                    rowvals, nzrange, nonzeros, spzeros

# import @reexport now to make it available for further imports/exports
using Reexport: @reexport

# Reexport all the Trixi.jl exports
@reexport using Trixi

# MPI needs to be imported before HDF5 to be able to use parallel HDF5
# as long as HDF5.jl uses Requires.jl to enable parallel HDF5 with MPI
using MPI: MPI

using SciMLBase: CallbackSet, DiscreteCallback,
                 ODEProblem, ODESolution, ODEFunction,
                 SplitODEProblem
import SciMLBase: get_du, get_tmp_cache, u_modified!,
                  AbstractODEIntegrator, init, step!, check_error,
                  get_proposed_dt, set_proposed_dt!,
                  terminate!, remake, add_tstop!, has_tstop, first_tstop

using CodeTracking: CodeTracking
using ConstructionBase: ConstructionBase
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
@reexport using EllipsisNotation # ..
using FillArrays: Ones, Zeros
using ForwardDiff: ForwardDiff
using HDF5: HDF5, h5open, attributes, create_dataset, datatype, dataspace
using IfElse: ifelse
using LinearMaps: LinearMap
using LoopVectorization: LoopVectorization, @turbo, indices
using StaticArrayInterface: static_length # used by LoopVectorization
using MuladdMacro: @muladd
using Octavian: Octavian, matmul!
using Polyester: Polyester, @batch # You know, the cheapest threads you can find...
using OffsetArrays: OffsetArray, OffsetVector
# using P4est
# using T8code
using Setfield: @set
using RecipesBase: RecipesBase
using Requires: @require
using Static: Static, One, True, False
@reexport using StaticArrays: SVector
using StaticArrays: StaticArrays, MVector, MArray, SMatrix, @SMatrix
using StrideArrays: PtrArray, StrideArray, StaticInt
@reexport using StructArrays: StructArrays, StructArray
using TimerOutputs: TimerOutputs, @notimeit, TimerOutput, print_timer, reset_timer!
using Triangulate: Triangulate, TriangulateIO, triangulate
export TriangulateIO # for type parameter in DGMultiMesh
using TriplotBase: TriplotBase
using TriplotRecipes: DGTriPseudocolor
@reexport using SimpleUnPack: @unpack
using SimpleUnPack: @pack!
using DataStructures: BinaryHeap, FasterForward, extract_all!

# RBFD.jl dependencies
import DynamicPolynomials: @polyvar
import DynamicPolynomials: monomials
import DynamicPolynomials: differentiate
# using DynamicPolynomials: @polyvar, monomials, differentiate
# using FixedPolynomials
using StaticPolynomials
using LinearAlgebra
using StaticArrays
# using HNSW
using SparseArrays
using DelimitedFiles
using HDF5
using Statistics
using WriteVTK
using Symbolics
using RuntimeGeneratedFunctions
using NearestNeighbors

# Define the entry points of our type hierarchy, e.g.
#     AbstractEquations, AbstractSemidiscretization etc.
# Placing them here allows us to make use of them for dispatch even for
# other stuff defined very early in our include pipeline, e.g.
#     IndicatorLöhner(semi::AbstractSemidiscretization)
include("basic_types.jl")

# Include all top-level source files
include("auxiliary/auxiliary.jl")
# include("auxiliary/mpi.jl")
# include("auxiliary/p4est.jl")
include("equations/equations.jl")
# include("meshes/meshes.jl")
include("domains/domains.jl")
include("solvers/solvers.jl")
# include("equations/equations_parabolic.jl") # these depend on parabolic solver types
# include("semidiscretization/semidiscretization.jl")
include("callbacks_step/callbacks_step.jl")
# include("callbacks_stage/callbacks_stage.jl")

# `trixi_include` and special elixirs such as `convergence_test`
# include("auxiliary/special_elixirs.jl")

# Plot recipes and conversion functions to visualize results with Plots.jl
# include("visualization/visualization.jl")

# export types/functions that define the public API of Trixi.jl

# Export Hyperbolic Equations
# export AcousticPerturbationEquations2D,
#        CompressibleEulerEquations1D, CompressibleEulerEquations2D,
#        CompressibleEulerEquations3D

# Export Parabolic Equations
# export LaplaceDiffusion1D, LaplaceDiffusion2D, LaplaceDiffusion3D,
#        CompressibleNavierStokesDiffusion1D, CompressibleNavierStokesDiffusion2D,
#        CompressibleNavierStokesDiffusion3D

# export GradientVariablesConservative, GradientVariablesPrimitive, GradientVariablesEntropy

# Export Flux Functions
# export flux, flux_central, flux_lax_friedrichs, flux_hll, flux_hllc, flux_hlle,
#        flux_godunov,
#        flux_chandrashekar, flux_ranocha, flux_derigs_etal, flux_hindenlang_gassner,
#        flux_nonconservative_powell, flux_nonconservative_powell_local_symmetric,
#        flux_kennedy_gruber, flux_shima_etal, flux_ec,
#        flux_fjordholm_etal, flux_nonconservative_fjordholm_etal,
#        flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal,
#        flux_es_ersing_etal, flux_nonconservative_ersing_etal,
#        flux_chan_etal, flux_nonconservative_chan_etal, flux_winters_etal,
#        hydrostatic_reconstruction_audusse_etal, flux_nonconservative_audusse_etal,
# # TODO: TrixiShallowWater: move anything with "chen_noelle" to new file
#        hydrostatic_reconstruction_chen_noelle, flux_nonconservative_chen_noelle,
#        flux_hll_chen_noelle,
#        FluxPlusDissipation, DissipationGlobalLaxFriedrichs, DissipationLocalLaxFriedrichs,
#        FluxLaxFriedrichs, max_abs_speed_naive,
#        FluxHLL, min_max_speed_naive, min_max_speed_davis, min_max_speed_einfeldt,
#        min_max_speed_chen_noelle,
#        FluxLMARS,
#        FluxRotated,
#        flux_shima_etal_turbo, flux_ranocha_turbo,
#        FluxHydrostaticReconstruction,
#        FluxUpwind

# Export Flux Splitting
# export splitting_steger_warming, splitting_vanleer_haenel,
#        splitting_coirier_vanleer, splitting_lax_friedrichs

# export initial_condition_constant,
#        initial_condition_gauss,
#        initial_condition_density_wave,
#        initial_condition_weak_blast_wave

# export boundary_condition_do_nothing,
#        boundary_condition_periodic,
#        BoundaryConditionDirichlet,
#        BoundaryConditionNeumann,
#        boundary_condition_noslip_wall,
#        boundary_condition_slip_wall,
#        boundary_condition_wall,
#        BoundaryConditionNavierStokesWall, NoSlip, Adiabatic, Isothermal,
#        BoundaryConditionCoupled

# export initial_condition_convergence_test, source_terms_convergence_test
# export source_terms_harmonic
# export initial_condition_poisson_nonperiodic, source_terms_poisson_nonperiodic,
#        boundary_condition_poisson_nonperiodic
# export initial_condition_eoc_test_coupled_euler_gravity,
#        source_terms_eoc_test_coupled_euler_gravity, source_terms_eoc_test_euler

# export cons2cons, cons2prim, prim2cons, cons2macroscopic, cons2state, cons2mean,
#        cons2entropy, entropy2cons
# export density, pressure, density_pressure, velocity, global_mean_vars,
#        equilibrium_distribution, waterheight_pressure
# export entropy, energy_total, energy_kinetic, energy_internal, energy_magnetic,
#        cross_helicity,
#        enstrophy
# export lake_at_rest_error
# export ncomponents, eachcomponent

# Export Mesh/Domain Types
export TreeMesh, StructuredMesh, UnstructuredMesh2D, P4estMesh, T8codeMesh

# Export Solvers and Methods
export DG,
       DGSEM, LobattoLegendreBasis,
       FDSBP,
       VolumeIntegralWeakForm, VolumeIntegralStrongForm,
       VolumeIntegralFluxDifferencing,
       VolumeIntegralPureLGLFiniteVolume,
       VolumeIntegralShockCapturingHG, IndicatorHennemannGassner,
# TODO: TrixiShallowWater: move new indicator
       IndicatorHennemannGassnerShallowWater,
       VolumeIntegralUpwind,
       SurfaceIntegralWeakForm, SurfaceIntegralStrongForm,
       SurfaceIntegralUpwind,
       MortarL2

export VolumeIntegralSubcellLimiting, BoundsCheckCallback,
       SubcellLimiterIDP, SubcellLimiterIDPCorrection

export nelements, nnodes, nvariables,
       eachelement, eachnode, eachvariable

# export SemidiscretizationHyperbolic, semidiscretize, compute_coefficients, integrate

# export SemidiscretizationHyperbolicParabolic

# export SemidiscretizationEulerAcoustics

# export SemidiscretizationEulerGravity, ParametersEulerGravity,
#        timestep_gravity_erk52_3Sstar!, timestep_gravity_carpenter_kennedy_erk54_2N!

# export SemidiscretizationCoupled

export SummaryCallback, SteadyStateCallback, AnalysisCallback, AliveCallback,
       SaveRestartCallback, SaveSolutionCallback, TimeSeriesCallback, VisualizationCallback,
       AveragingCallback,
       AMRCallback, StepsizeCallback,
       GlmSpeedCallback, LBMCollisionCallback, EulerAcousticsCouplingCallback,
       TrivialCallback, AnalysisCallbackCoupled

export load_mesh, load_time, load_timestep, load_timestep!, load_dt,
       load_adaptive_time_integrator!

# export ControllerThreeLevel, ControllerThreeLevelCombined,
#        IndicatorLöhner, IndicatorLoehner, IndicatorMax

# # TODO: TrixiShallowWater: move new limiter
# export PositivityPreservingLimiterZhangShu, PositivityPreservingLimiterShallowWater

export trixi_include, examples_dir, get_examples, default_example,
       default_example_unstructured, ode_default_options

export ode_norm, ode_unstable_check

export convergence_test, jacobian_fd, jacobian_ad_forward, linear_structure

export DGMulti, DGMultiBasis, estimate_dt, DGMultiMesh, GaussSBP

export ViscousFormulationBassiRebay1, ViscousFormulationLocalDG

# Visualization-related exports
# export PlotData1D, PlotData2D, ScalarPlotData2D, getmesh, adapt_to_mesh_level!,
#        adapt_to_mesh_level,
#        iplot, iplot!

# include("auxiliary/precompile.jl")
# _precompile_manual_()

end
