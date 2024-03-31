using Test
using MeshfreeTrixi
# using SafeTestsets

@testset "First Order Operator Test" begin
    include("first_order_test.jl")
end

# # Original Poisson Test
# @safetestset "First Order Operator Test" begin
#     include("first_order_test.jl")
# end

# # Poisson Test w/ Mesh Import
# @safetestset "Hyperviscosity Tominec Test" begin
#     include("hv_tom_test.jl")
# end

# # Hyperviscosity Operator Test
# @safetestset "Artificial Viscosity Test" begin
#     include("av_test.jl")
# end