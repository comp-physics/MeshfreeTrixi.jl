# Based on Trixi/src/basic_types.jl
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# abstract supertype of specific semidiscretizations such as
# - SemidiscretizationHyperbolic for hyperbolic conservation laws
# - SemidiscretizationEulerGravity for Euler with self-gravity
abstract type AbstractSemidiscretization end

"""
    AbstractDomain{NDIMS}

An abstract supertype of specific mesh types such as `TreeMesh` or `StructuredMesh`.
The type parameters encode the number of spatial dimensions (`NDIMS`).
"""
abstract type AbstractDomain{NDIMS} end
end # @muladd
