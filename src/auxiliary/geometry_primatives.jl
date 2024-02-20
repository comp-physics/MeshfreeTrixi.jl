# NodesandModes for PointCloudDomain
# each type of element shape - used for dispatch only (basis specifically)
abstract type AbstractElemShape{NDIMS} end
struct Point1D <: AbstractElemShape{1} end
struct Point2D <: AbstractElemShape{2} end
struct Point3D <: AbstractElemShape{3} end

dimensionality(elem::AbstractElemShape{Dim}) where {Dim} = Dim

# StartUpDG for PointCloudDomain. Uses base types from NodesAndModes
# May rename RefElemData to RefPointData to prevent confusion 
# or conflicts with RefElemData in StartUpDG
"""
    struct RefElemData

RefElemData: contains info (point coords, neighbors, order of accuracy)
for a high order RBF basis on a given reference element. 

Example:
```julia
N = 3
rd = RefElemData(Tri(), N)
(; r, s ) = rd
```
"""
### RefElemData called by basis
# Rework for PointCloudDomain
struct RefElemData{Dim, ElemShape <: AbstractElemShape{Dim}, ApproximationType,
                   NT, NV, F}
    element_type::ElemShape
    approximation_type::ApproximationType # Polynomial / SBP{...}

    N::NT               # polynomial degree of accuracy
    nv::NV               # number of neighbors
    f::F               # basis function 
end

# need this to use @set outside of StartUpDG
function ConstructionBase.setproperties(rd::RefElemData, patch::NamedTuple)
    fields = (haskey(patch, symbol) ? getproperty(patch, symbol) : getproperty(rd, symbol) for symbol in fieldnames(typeof(rd)))
    return RefElemData(fields...)
end

function ConstructionBase.getproperties(rd::RefElemData)
    (; element_type = rd.element_type, approximation_type = rd.approximation_type, N = rd.N,
     nv = rd.nv,
     f = rd.f)
end

# Updated _propertynames function to reflect the new fields in RefElemData
_propertynames(::Type{RefElemData}, private::Bool = false) = (:nv, :f)

function Base.propertynames(x::RefElemData, private::Bool = false)
    return (fieldnames(typeof(x))..., _propertynames(typeof(x))...)
end

# convenience unpacking routines
# Not necessary for PointCloudDomain
# function Base.getproperty(x::RefElemData{Dim, ElementType, ApproxType},
#                           s::Symbol) where {Dim, ElementType, ApproxType}
#     return getfield(x, s)
# end

"""
    function RefElemData(elem; N, kwargs...)
    function RefElemData(elem, approx_type; N, kwargs...)

Keyword argument constructor for RefElemData (to "label" `N` via `rd = RefElemData(Line(), N=3)`)
"""
RefElemData(elem; N, kwargs...) = RefElemData(elem, N; kwargs...)
RefElemData(elem, approx_type; N, kwargs...) = RefElemData(elem, approx_type, N; kwargs...)

# default to Polynomial-type RefElemData
RefElemData(elem, N::Int; kwargs...) = RefElemData(elem, RBF(), N; kwargs...)

@inline Base.ndims(::Point1D) = 1
@inline Base.ndims(::Point2D) = 2
@inline Base.ndims(::Point3D) = 3

# ====================================================
#          RefElemData approximation types
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
# RBF(rbf_type::DefaultRBFType, Nrbf::Int = 3) = RBF{DefaultRBFType}(rbf_type, Nrbf)
RBF(rbf_type::DefaultRBFType) = RBF{DefaultRBFType}(rbf_type, rbf_type.Nrbf)

# RBF(PolyharmonicSpline()) type indicates (N+1)-point Gauss quadrature on tensor product elements
struct PolyharmonicSpline
    Nrbf::Int
    PolyharmonicSpline(Nrbf::Int = 3) = new(Nrbf)  # Default order is 3
end
# RBF{PolyharmonicSpline}() = RBF(PolyharmonicSpline())
# RBF(rbf_type::PolyharmonicSpline, Nrbf::Int = 3) = RBF{PolyharmonicSpline}(rbf_type, Nrbf)
RBF(rbf_type::PolyharmonicSpline) = RBF{PolyharmonicSpline}(rbf_type, rbf_type.Nrbf)

# ====================================
#              Printing 
# ====================================

function Base.show(io::IO, ::MIME"text/plain", rd::RefElemData)
    @nospecialize rd
    print(io,
          "RefElemData for a degree $(rd.N) approximation \nusing degree $(rd.approximation_type.Nrbf) $(_short_typeof(rd.approximation_type))s " *
          "on a $(_short_typeof(rd.element_type)) element.")
end

function Base.show(io::IO, rd::RefElemData)
    @nospecialize basis # reduce precompilation time
    print(io,
          "RefElemData{N=$(rd.N), $(_short_typeof(rd.approximation_type)), $(_short_typeof(rd.element_type))}.")
end

_short_typeof(x) = typeof(x)

_short_typeof(approx_type::RBF{<:DefaultRBFType}) = "RBF"
_short_typeof(approx_type::RBF{<:PolyharmonicSpline}) = "RBF{PolyharmonicSpline}"
# function _short_typeof(approx_type::Polynomial{<:TensorProductQuadrature})
#     "Polynomial{TensorProductQuadrature}"
# end

"""
    RefElemData(elem::Line, N;
                quad_rule_vol = quad_nodes(elem, N+1))
    RefElemData(elem::Union{Tri, Quad}, N;
                 quad_rule_vol = quad_nodes(elem, N),
                 quad_rule_face = gauss_quad(0, 0, N))
    RefElemData(elem::Union{Hex, Tet}, N;
                 quad_rule_vol = quad_nodes(elem, N),
                 quad_rule_face = quad_nodes(Quad(), N))
    RefElemData(elem; N, kwargs...) # version with keyword args

Constructor for `RefElemData` for different element types.
"""
# Default RefElemData
function RefElemData(elem::Union{Point1D, Point2D, Point3D},
                     approx_type::RBF{DefaultRBFType}, N)
    # Construct basis functions on reference element
    # Default to PolyharmonicSpline RBFs w/ appended polynomials
    Nrbf = approx_type.Nrbf
    approx_type = RBF(PolyharmonicSpline(Nrbf))
    F = nothing

    # Number of neighbors
    d = dimensionality(elem)
    min_NV = [10, 15, 20]
    NV = max(2 * binomial(N + d, d), min_NV[d])

    return RefElemData(elem, approx_type, N, NV, F)
end

function RefElemData(elem::Union{Point1D, Point2D, Point3D},
                     approx_type::RBF, N)
    # Construct basis functions on reference element
    # Default to PolyharmonicSpline RBFs w/ appended polynomials
    F = nothing

    # Number of neighbors
    d = dimensionality(elem)
    min_NV = [10, 15, 20]
    NV = max(2 * binomial(N + d, d), min_NV[d])

    return RefElemData(elem, approx_type, N, NV, F)
end