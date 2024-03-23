# The following statements below outside the `@muladd begin ... end` block, as otherwise
# Revise.jl might be broken

include("medusa/read_medusa_file.jl")

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Print informative message at startup
# function Trixi.print_startup_message()
#     s = """

#       ████████╗█████╗  █████╗███████╗
#       ╚══██╔══╝██║╚██╗██╔╝██║██╔════╝
#          ██║   ██║ ╚███╔╝ ██║██████╗
#          ██║   ██║   ╚═╝  ██║██╔═══╝
#          ██║   ██║        ██║██║ 
#          ╚═╝   ╚═╝        ╚═╝╚═╝ 
#       """
#     mpi_println(s)
# end
end # @muladd
