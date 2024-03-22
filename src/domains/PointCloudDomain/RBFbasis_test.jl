using Revise
using Trixi
using ConstructionBase
using MuladdMacro
using Trixi: @threaded
using Trixi: @trixi_timeit
using Trixi: summary_header, summary_line, summary_footer, increment_indent
using Trixi: True, False
using NearestNeighbors
using LinearAlgebra
using SparseArrays
using StructArrays
# includet("geometry_primatives.jl")
includet("../header.jl")

# Base Methods
basis = RefPointData(Point1D(), RBF(DefaultRBFType(5)), 5)
basis = RefPointData(Point2D(), RBF(), 5)
basis = RefPointData(Point1D(), RBF(PolyharmonicSpline(5)), 3)
rbf = RBFSolver(basis, RBFFDEngine())

# Specialized Methods
basis = PointCloudBasis(Point3D(), 3; approximation_type = RBF(PolyharmonicSpline(5)))
basis = PointCloudBasis(Point2D(), 3; approximation_type = RBF(PolyharmonicSpline(5)))
