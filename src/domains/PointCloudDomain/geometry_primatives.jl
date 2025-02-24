# Based on NodesandModes for PointCloudDomain
# each type of element shape - used for dispatch only (basis specifically)
abstract type AbstractElemShape{NDIMS} end
struct Point1D <: AbstractElemShape{1} end
struct Point2D <: AbstractElemShape{2} end
struct Point3D <: AbstractElemShape{3} end

dimensionality(elem::AbstractElemShape{Dim}) where {Dim} = Dim

# Based on StartUpDG for PointCloudDomain. Uses base types from NodesAndModes
# May rename RefPointData to RefPointData to prevent confusion 
# or conflicts with RefPointData in StartUpDG
"""
    struct RefPointData

RefPointData: contains info (point coords, neighbors, order of accuracy)
for a high order RBF basis on a given reference element. 

Example:
```julia
N = 3
rd = RefPointData(Tri(), N)
(; r, s ) = rd
```
"""
### RefPointData called by basis
struct RefPointData{Dim, ElemShape <: AbstractElemShape{Dim}, ApproximationType,
                    NT, NV, F}
    element_type::ElemShape
    approximation_type::ApproximationType # RBF / PHS{...}

    N::NT              # polynomial degree of accuracy
    nv::NV             # number of neighbors
    f::F               # basis function 
end

# need this to use @set outside of StartUpDG
function ConstructionBase.setproperties(rd::RefPointData, patch::NamedTuple)
    fields = (haskey(patch, symbol) ? getproperty(patch, symbol) : getproperty(rd, symbol) for symbol in fieldnames(typeof(rd)))
    return RefPointData(fields...)
end

function ConstructionBase.getproperties(rd::RefPointData)
    (; element_type = rd.element_type, approximation_type = rd.approximation_type, N = rd.N,
     nv = rd.nv,
     f = rd.f)
end

# # Updated _propertynames function to reflect the new fields in RefPointData
# _propertynames(::Type{RefPointData}, private::Bool = false) = (:nv, :f)

# function Base.propertynames(x::RefPointData, private::Bool = false)
#     return (fieldnames(typeof(x))..., _propertynames(typeof(x))...)
# end

function Base.propertynames(x::RefPointData, private::Bool = false)
    return fieldnames(typeof(x))
end

# convenience unpacking routines
# Not necessary for PointCloudDomain
# function Base.getproperty(x::RefPointData{Dim, ElementType, ApproxType},
#                           s::Symbol) where {Dim, ElementType, ApproxType}
#     return getfield(x, s)
# end

"""
    function RefPointData(elem; N, kwargs...)
    function RefPointData(elem, approx_type; N, kwargs...)

Keyword argument constructor for RefPointData (to "label" `N` via `rd = RefPointData(Line(), N=3)`)
"""
RefPointData(elem; N, kwargs...) = RefPointData(elem, N; kwargs...)
function RefPointData(elem, approx_type; N, kwargs...)
    RefPointData(elem, approx_type, N; kwargs...)
end

# default to RBF-type RefPointData
RefPointData(elem, N::Int; kwargs...) = RefPointData(elem, RBF(), N; kwargs...)

@inline Base.ndims(::Point1D) = 1
@inline Base.ndims(::Point2D) = 2
@inline Base.ndims(::Point3D) = 3

# ====================================================
#          RefPointData approximation types
# ====================================================

"""
    RBF{T}

Represents RBF approximation types (as opposed to generic polynomials). 
By default, `RBF()` constructs a `RBF{DefaultRBFType}`.
Specifying a type parameters allows for dispatch on additional structure within an
RBF approximation (e.g., polyharmonic spline, gaussian, etc). 
"""
struct RBF{T}
    rbf_type::T
    Nrbf::Int  # Order for the RBF
end

struct DefaultRBFType
    Nrbf::Int
    DefaultRBFType(Nrbf::Int = 3) = new(Nrbf)  # Default order is 3
end
RBF() = RBF(DefaultRBFType())
RBF(rbf_type::DefaultRBFType) = RBF{DefaultRBFType}(rbf_type, rbf_type.Nrbf)

# RBF(PolyharmonicSpline()) type indicates odd order polyharmonic spline RBFs w/ appended monomials
struct PolyharmonicSpline
    Nrbf::Int
    # PolyharmonicSpline(Nrbf::Int = 3) = new(Nrbf)  # Default order is 3
end
PolyharmonicSpline(; Nrbf::Int = 3) = PolyharmonicSpline(Nrbf)
RBF(rbf_type::PolyharmonicSpline) = RBF{PolyharmonicSpline}(rbf_type, rbf_type.Nrbf)

# RBF(HybridGaussianPHS()) type indicates blended Gaussian and odd order polyharmonic spline RBFs w/ appended monomials
struct HybridGaussianPHS
    Nrbf::Int
    alpha::Float64
    beta::Float64
    epsilon::Float64
    # function HybridGaussianPHS(Nrbf::Int=3, alpha::Float64=1.0, beta::Float64=1.0,
    #     epsilon::Float64=1.0)
    #     new(Nrbf, alpha, beta, epsilon)
    # end  # Default order is 3
end
function HybridGaussianPHS(; Nrbf::Int = 3, alpha::Float64 = 1.0, beta::Float64 = 1.0,
                           epsilon::Float64 = 1.0)
    HybridGaussianPHS(Nrbf, alpha, beta, epsilon)
end
RBF(rbf_type::HybridGaussianPHS) = RBF{HybridGaussianPHS}(rbf_type, rbf_type.Nrbf)

# ====================================
#              Printing 
# ====================================

function Base.show(io::IO, ::MIME"text/plain", rd::RefPointData)
    @nospecialize rd
    print(io,
          "RefPointData for a degree $(rd.N) approximation \nusing degree $(rd.approximation_type.Nrbf) $(_short_typeof(rd.approximation_type))s " *
          "on a $(_short_typeof(rd.element_type)) element.")
end

function Base.show(io::IO, rd::RefPointData)
    @nospecialize basis # reduce precompilation time
    print(io,
          "RefPointData{N=$(rd.N), $(_short_typeof(rd.approximation_type)), $(_short_typeof(rd.element_type))}.")
end

_short_typeof(x) = typeof(x)

_short_typeof(approx_type::RBF{<:DefaultRBFType}) = "RBF"
_short_typeof(approx_type::RBF{<:PolyharmonicSpline}) = "RBF{PolyharmonicSpline}"
# function _short_typeof(approx_type::RBF{<:TensorProductQuadrature})
#     "RBF{TensorProductQuadrature}"
# end

"""
    RefPointData(elem::Line, N;
                quad_rule_vol = quad_nodes(elem, N+1))
    RefPointData(elem::Union{Tri, Quad}, N;
                 quad_rule_vol = quad_nodes(elem, N),
                 quad_rule_face = gauss_quad(0, 0, N))
    RefPointData(elem::Union{Hex, Tet}, N;
                 quad_rule_vol = quad_nodes(elem, N),
                 quad_rule_face = quad_nodes(Quad(), N))
    RefPointData(elem; N, kwargs...) # version with keyword args

Constructor for `RefPointData` for different element types.
"""
# Default RefPointData
function RefPointData(elem::AbstractElemShape{Dim},
                      approx_type::RBF{DefaultRBFType}, N) where {Dim}
    # Construct basis functions on reference element
    # Default to PolyharmonicSpline RBFs w/ appended polynomials
    Nrbf = approx_type.Nrbf
    approx_type = RBF(PolyharmonicSpline(Nrbf))
    F = create_basis(elem, approx_type, N)

    # Number of neighbors
    d = dimensionality(elem)
    min_NV = [10, 15, 20]
    NV = max(2 * binomial(N + d, d), min_NV[d])

    return RefPointData(elem, approx_type, N, NV, F)
end

function RefPointData(elem::AbstractElemShape{Dim},
                      approx_type::RBF, N) where {Dim}
    # Construct basis functions on reference element
    # Default to PolyharmonicSpline RBFs w/ appended polynomials
    F = create_basis(elem, approx_type, N)

    # Number of neighbors
    # d = dimensionality(elem)
    min_NV = [10, 15, 20]
    NV = max(2 * binomial(N + Dim, Dim), min_NV[Dim])

    return RefPointData(elem, approx_type, N, NV, F)
end

function create_basis(elem::AbstractElemShape{Dim},
                      approx_type::RBF, N) where {Dim}
    rbf = rbf_basis(elem, approx_type, N)
    poly = poly_basis(elem, approx_type, N)

    return (; rbf, poly)
end

# PHS
function rbf_basis(elem::AbstractElemShape{1},
                   approx_type::RBF{PolyharmonicSpline}, N)
    # Specialize this function to create RBF bases for specific
    # RBF types and dimensions
    p = approx_type.Nrbf
    @variables x
    rbf = sqrt(x^2)^p
    rbf_expr = build_function(rbf, [x]; expression = Val{false})
    return rbf
end
function rbf_basis(elem::AbstractElemShape{2},
                   approx_type::RBF{PolyharmonicSpline}, N)
    p = approx_type.Nrbf
    @variables x y
    rbf = sqrt(x^2 + y^2)^p
    rbf_expr = build_function(rbf, [x, y]; expression = Val{false})
    return rbf
end
function rbf_basis(elem::AbstractElemShape{3},
                   approx_type::RBF{PolyharmonicSpline}, N)
    p = approx_type.Nrbf
    @variables x y z
    rbf = sqrt(x^2 + y^2 + z^2)^p
    rbf_expr = build_function(rbf, [x, y, z]; expression = Val{false})
    return rbf
end
# Hybrid Kernel
function rbf_basis(elem::AbstractElemShape{1},
                   approx_type::RBF{HybridGaussianPHS}, N)
    # Specialize this function to create RBF bases for specific
    # RBF types and dimensions
    @unpack Nrbf, alpha, beta, epsilon = approx_type.rbf_type
    @variables x
    r = sqrt(x^2)
    rbf = alpha * exp(-(epsilon * r)^2) + beta * r^Nrbf
    rbf_expr = build_function(rbf, [x]; expression = Val{false})
    return rbf
end
function rbf_basis(elem::AbstractElemShape{2},
                   approx_type::RBF{HybridGaussianPHS}, N)
    @unpack Nrbf, alpha, beta, epsilon = approx_type.rbf_type
    @variables x y
    r = sqrt(x^2 + y^2)
    rbf = alpha * exp(-(epsilon * r)^2) + beta * r^Nrbf
    rbf_expr = build_function(rbf, [x, y]; expression = Val{false})
    return rbf
end
function rbf_basis(elem::AbstractElemShape{3},
                   approx_type::RBF{HybridGaussianPHS}, N)
    @unpack Nrbf, alpha, beta, epsilon = approx_type.rbf_type
    @variables x y z
    r = sqrt(x^2 + y^2 + z^2)
    rbf = alpha * exp(-(epsilon * r)^2) + beta * r^Nrbf
    rbf_expr = build_function(rbf, [x, y, z]; expression = Val{false})
    return rbf
end

function poly_basis(elem::AbstractElemShape{1},
                    approx_type::RBF, N)
    # Specialize this function to create polynomial bases for specific
    # RBF types
    @polyvar x
    poly = monomials([x], 0:N)
    return poly
end
function poly_basis(elem::AbstractElemShape{2},
                    approx_type::RBF, N)
    # Specialize this function to create polynomial bases for specific
    # RBF types
    @polyvar x y
    poly = monomials([x, y], 0:N)
    return poly
end
function poly_basis(elem::AbstractElemShape{3},
                    approx_type::RBF, N)
    # Specialize this function to create polynomial bases for specific
    # RBF types
    @polyvar x y z
    poly = monomials([x, y, z], 0:N)
    return poly
end

# Similar to StartUpDG.MeshData
# This is the underlying data structure for PointCloudDomain
# Mostly reworked 
struct PointData{Dim, Tv, Ti}
    points::Vector{Tv}                # Point coordinates
    neighbors::Vector{Vector{Ti}}     # Neighbors for each point
    num_points::Int                   # Number of points
    num_neighbors::Int                # Number of neighbors lists
    dx_min::Float64                   # Minimum distance between points
    dx_avg::Float64                   # Average distance between points
end

## We need to calculate dx_min at runtime so we cannot use this constructor
# function PointData(points::Vector{Tv},
#                    neighbors::Vector{Vector{Ti}}) where {
#                                                          Dim,
#                                                          Tv <: SVector{Dim, Float64},
#                                                          Ti
#                                                          }
#     PointData{Dim, Tv, Ti}(points, neighbors, length(points), length(neighbors[1]))
# end

"""
    PointData(medusa_data::Vector{Tv}, basis) where {Tv <: SVector{Dim, Float64}}

- `medusa_data` contains point positions for entire point cloud.
- `basis` contains all basis information including the number of required neighbors to support required order of accuracy.
"""
function PointData(medusa_data::Vector{Tv},
                   basis::RefPointData) where {Dim, Tv <: SVector{Dim, Float64}}
    nv = basis.nv  # The number of neighbors

    # Calculate neighbor list for all points
    kdtree = KDTree(medusa_data)
    n_idxs, n_dists = knn(kdtree, medusa_data, nv, true)
    dx_min = minimum(n_dists)

    # Calculate avg and min distances between points
    idxs_y, dists_y = knn(kdtree, medusa_data, 2, true)
    dx_avg = mean(dists_y)[2]
    dx_min = minimum(dists_y)[2]

    # Instantiate PointData with the points and neighbors. The num_points and num_neighbors are automatically computed.
    return PointData{Dim, Tv, Int}(medusa_data, n_idxs, length(medusa_data), nv, dx_min,
                                   dx_avg)
end

function PointData(medusa_data::Vector{Tv},
                   basis::RefPointData, dx_min::Float64,
                   dx_avg::Float64) where {Dim, Tv <: SVector{Dim, Float64}}
    nv = basis.nv  # The number of neighbors

    # Calculate neighbor list for all points
    kdtree = KDTree(medusa_data)
    n_idxs, n_dists = knn(kdtree, medusa_data, nv, true)
    # dx_min = minimum(n_dists)

    # Calculate avg and min distances between points
    idxs_y, dists_y = knn(kdtree, medusa_data, 2, true)
    # dx_avg = mean(dists_y)[2]
    # dx_min = minimum(dists_y)[2]

    # Instantiate PointData with the points and neighbors. The num_points and num_neighbors are automatically computed.
    return PointData{Dim, Tv, Int}(medusa_data, n_idxs, length(medusa_data), nv, dx_min,
                                   dx_avg)
end

struct BoundaryData{Ti <: Integer, Tv <: SVector{N, T} where {N, T <: Number}}
    idx::Vector{Ti}       # Indices of boundary points
    normals::Vector{Tv}   # Normals at boundary points
end
