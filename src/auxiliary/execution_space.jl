# Supertype for specializing methods for execution on different platforms
# Instantiated when creating solver object
# i.e.  execution_space = CUDAExecutionSpace()
#       solver = PointCloudSolver(basis; execution_space = execution_space)

abstract type ExecutionSpace end
struct CPUExecutionSpace <: ExecutionSpace end
struct CUDAExecutionSpace <: ExecutionSpace end