# Supertype for specializing methods for execution on different platforms
# Instantiated when creating solver object
# i.e.  execution_space = CUDAExecutionSpace()
#       solver = PointCloudSolver(basis; execution_space = execution_space)

abstract type ExecutionSpace end
struct CPUExecutionSpace <: ExecutionSpace end
struct CUDAExecutionSpace <: ExecutionSpace end

function wrap_array_exec_space(array::ArrayType, space::CPUExecutionSpace) where {ArrayType}
    return array
end

function wrap_array_exec_space(array::Vector, space::CUDAExecutionSpace)
    return CuArray(array)
end

function wrap_array_exec_space(array::Matrix, space::CUDAExecutionSpace)
    return CuArray(array)
end

function wrap_array_exec_space(array::SparseMatrixCSC, space::CUDAExecutionSpace)
    return CuSparseMatrixCSC(array)
end