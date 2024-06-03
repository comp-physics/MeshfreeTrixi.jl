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

# using LinearAlgebra: LinearAlgebra, Diagonal, diag, dot, mul!, norm, cross, normalize, I,
#                      UniformScaling, det
# using Printf: @printf, @sprintf, println
# using SparseArrays: AbstractSparseMatrix, AbstractSparseMatrixCSC, sparse, droptol!,
#                     rowvals, nzrange, nonzeros, spzeros

# import @reexport now to make it available for further imports/exports
using Reexport: @reexport

# Reexport all the Trixi.jl exports
@reexport using Trixi

using Trixi: @threaded
using Trixi: @trixi_timeit, timer
# using Trixi: summary_header, summary_line, summary_footer, increment_indent, summary_box
using Trixi: True, False, nvariables, have_nonconservative_terms, One

import Trixi: rhs!

# Trixi Dependencies
# # MPI needs to be imported before HDF5 to be able to use parallel HDF5
# # as long as HDF5.jl uses Requires.jl to enable parallel HDF5 with MPI
# using MPI: MPI

# using SciMLBase: CallbackSet, DiscreteCallback,
#                  ODEProblem, ODESolution, ODEFunction,
#                  SplitODEProblem
# import SciMLBase: get_du, get_tmp_cache, u_modified!,
#                   AbstractODEIntegrator, init, step!, check_error,
#                   get_proposed_dt, set_proposed_dt!,
#                   terminate!, remake, add_tstop!, has_tstop, first_tstop

# using CodeTracking: CodeTracking
# using ConstructionBase: ConstructionBase
# using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
# @reexport using EllipsisNotation # ..
# using FillArrays: Ones, Zeros
# using ForwardDiff: ForwardDiff
# using HDF5: HDF5, h5open, attributes, create_dataset, datatype, dataspace
# using IfElse: ifelse
# using LinearMaps: LinearMap
# using LoopVectorization: LoopVectorization, @turbo, indices
# using StaticArrayInterface: static_length # used by LoopVectorization
# using MuladdMacro: @muladd
# using Polyester: Polyester, @batch # You know, the cheapest threads you can find...
# using OffsetArrays: OffsetArray, OffsetVector
# # using P4est
# # using T8code
# using Setfield: @set
# using RecipesBase: RecipesBase
# using Requires: @require
# using Static: Static, One, True, False
# @reexport using StaticArrays: SVector
# using StaticArrays: StaticArrays, MVector, MArray, SMatrix, @SMatrix
# using StrideArrays: PtrArray, StrideArray, StaticInt
# @reexport using StructArrays: StructArrays, StructArray
# using TimerOutputs: TimerOutputs, @notimeit, TimerOutput, print_timer, reset_timer!
# using Triangulate: Triangulate, TriangulateIO, triangulate
# export TriangulateIO # for type parameter in DGMultiMesh
# using TriplotBase: TriplotBase
# using TriplotRecipes: DGTriPseudocolor
# @reexport using SimpleUnPack: @unpack
# using SimpleUnPack: @pack!
# using DataStructures: BinaryHeap, FasterForward, extract_all!

# RBFD.jl dependencies
using SciMLBase: CallbackSet, DiscreteCallback,
                 ODEProblem, ODESolution, ODEFunction,
                 SplitODEProblem
import SciMLBase: get_du, get_tmp_cache, u_modified!,
                  AbstractODEIntegrator, init, step!, check_error,
                  get_proposed_dt, set_proposed_dt!,
                  terminate!, remake, add_tstop!, has_tstop, first_tstop
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
using DelimitedFiles
import DynamicPolynomials: @polyvar
import DynamicPolynomials: monomials
import DynamicPolynomials: differentiate
# using StaticPolynomials
using StaticPolynomials: StaticPolynomials, evaluate, Polynomial, PolynomialSystem
using Symbolics
# using Trixi
using ConstructionBase
using MuladdMacro
using NearestNeighbors
using LinearAlgebra
using SparseArrays
using StructArrays
using Statistics
using RecursiveArrayTools: recursivecopy!
using OrdinaryDiffEq
using Octavian: Octavian, matmul!
using Polyester
using TimerOutputs: TimerOutputs, @notimeit, TimerOutput, print_timer, reset_timer!
@reexport using SimpleUnPack: @unpack
using SimpleUnPack: @pack!
using NaNMath
using Printf: @printf, @sprintf
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes, paraview_collection, vtk_save
using IterativeSolvers
import Base: *
import LinearAlgebra: mul!
using RadialBasisFiniteDifferences

# Define the entry points of our type hierarchy, e.g.
#     AbstractEquations, AbstractSemidiscretization etc.
# Placing them here allows us to make use of them for dispatch even for
# other stuff defined very early in our include pipeline, e.g.
#     IndicatorLöhner(semi::AbstractSemidiscretization)
include("basic_types.jl")

# Include all top-level source files
include("auxiliary/auxiliary.jl")
# include("auxiliary/geometry_primatives.jl")
# include("auxiliary/mpi.jl")
# include("auxiliary/p4est.jl")
# include("equations/equations.jl") 
include("equations/PointCloudBCs.jl")
include("domains/domains.jl")
include("solvers/solvers.jl")
# include("equations/equations_parabolic.jl") # these depend on parabolic solver types
# include("semidiscretization/semidiscretization.jl")
include("sources/generic_sources.jl")
include("callbacks_step/callbacks_step.jl")
include("callbacks_stage/callbacks_stage.jl")

# `trixi_include` and special elixirs such as `convergence_test`
# include("auxiliary/special_elixirs.jl")

# Plot recipes and conversion functions to visualize results with Plots.jl
include("visualization/visualization.jl")

# export types/functions that define the public API of Trixi.jl

# Export Hyperbolic Equations

# Export Flux Functions

# Export Initial Conditions

# Export Boundary Conditions
export BoundaryConditionDoNothing

# Export Mesh/Domain Types
export PointCloudDomain

# Export Solvers and Methods
# Engines replace VolumeIntegral
export PointCloudSolver, RBFSolver,
       RBFFDEngine

# Internal Methods for Solvers
export concrete_rbf_flux_basis, concrete_poly_flux_basis, compute_flux_operator

# export nelements, nnodes, nvariables,
#        eachelement, eachnode, eachvariable

# Export Basis Details
export RefPointData, Point1D, Point2D, Point3D,
       PointCloudBasis, RBF, PolyharmonicSpline, HybridGaussianPHS

export HistoryCallback, InfoCallback, SolutionSavingCallback, PerformanceCallback

export SourceTerms, SourceHyperviscosityFlyer, SourceHyperviscosityTominec,
       SourceUpwindViscosityTominec,
       SourceResidualViscosityTominec,
       SourceIGR

# Visualization-related exports
# export PlotData1D, PlotData2D, ScalarPlotData2D, getmesh, adapt_to_mesh_level!,
#        adapt_to_mesh_level,
#        iplot, iplot!

# include("auxiliary/precompile.jl")
# _precompile_manual_()

end
