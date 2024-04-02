function concrete_rbf_flux_basis(rbf, basis::RefPointData{NDIMS}) where {NDIMS}
    if NDIMS == 1
        @variables x
        Dx = Differential(x)
        rbf_x = simplify(expand_derivatives(Dx(rbf)))
        rbf_expr = build_function(rbf, [x]; expression = Val{false})
        rbf_x_expr = build_function(rbf_x, [x, y]; expression = Val{false})
        return (; rbf_expr, rbf_x_expr)
    elseif NDIMS == 2
        @variables x y
        Dx = Differential(x)
        Dy = Differential(y)
        rbf_x = simplify(expand_derivatives(Dx(rbf)))
        rbf_y = simplify(expand_derivatives(Dy(rbf)))
        rbf_expr = build_function(rbf, [x, y]; expression = Val{false})
        rbf_x_expr = build_function(rbf_x, [x, y]; expression = Val{false})
        rbf_y_expr = build_function(rbf_y, [x, y]; expression = Val{false})
        return (; rbf_expr, rbf_x_expr, rbf_y_expr)
    elseif NDIMS == 3
        @variables x y z
        Dx = Differential(x)
        Dy = Differential(y)
        Dz = Differential(z)
        rbf_x = simplify(expand_derivatives(Dx(rbf)))
        rbf_y = simplify(expand_derivatives(Dy(rbf)))
        rbf_z = simplify(expand_derivatives(Dz(rbf)))
        rbf_expr = build_function(rbf, [x, y, z]; expression = Val{false})
        rbf_x_expr = build_function(rbf_x, [x, y]; expression = Val{false})
        rbf_y_expr = build_function(rbf_y, [x, y]; expression = Val{false})
        rbf_z_expr = build_function(rbf_z, [x, y]; expression = Val{false})
        return (; rbf_expr, rbf_x_expr, rbf_y_expr, rbf_z_expr)
    end
end

# Specialized Basis generation for k-th derivative order
function concrete_rbf_flux_basis(rbf, basis::RefPointData{NDIMS}, k::Int) where {NDIMS}
    if NDIMS == 1
        @variables x
        Dxk = Differential(x)^k
        rbf_xk = simplify(expand_derivatives(Dxk(rbf)))
        rbf_expr = build_function(rbf, [x, y]; expression = Val{false})
        rbf_xk_expr = build_function(rbf_xk, [x, y]; expression = Val{false})
        return (; rbf_expr,
                rbf_x_expr = rbf_xk_expr)
    elseif NDIMS == 2
        @variables x y
        Dxk = Differential(x)^k
        Dyk = Differential(y)^k
        rbf_xk = simplify(expand_derivatives(Dxk(rbf)))
        rbf_yk = simplify(expand_derivatives(Dyk(rbf)))
        rbf_expr = build_function(rbf, [x, y]; expression = Val{false})
        rbf_xk_expr = build_function(rbf_xk, [x, y]; expression = Val{false})
        rbf_yk_expr = build_function(rbf_yk, [x, y]; expression = Val{false})
        return (; rbf_expr,
                rbf_x_expr = rbf_xk_expr,
                rbf_y_expr = rbf_yk_expr)
    elseif NDIMS == 3
        @variables x y z
        Dxk = Differential(x)^k
        Dyk = Differential(y)^k
        Dzk = Differential(z)^k
        rbf_xk = simplify(expand_derivatives(Dxk(rbf)))
        rbf_yk = simplify(expand_derivatives(Dyk(rbf)))
        rbf_zk = simplify(expand_derivatives(Dzk(rbf)))
        rbf_expr = build_function(rbf, [x, y, z]; expression = Val{false})
        rbf_xk_expr = build_function(rbf_xk, [x, y, z]; expression = Val{false})
        rbf_yk_expr = build_function(rbf_yk, [x, y, z]; expression = Val{false})
        rbf_zk_expr = build_function(rbf_zk, [x, y, z]; expression = Val{false})
        return (; rbf_expr,
                rbf_x_expr = rbf_xk_expr,
                rbf_y_expr = rbf_yk_expr,
                rbf_z_expr = rbf_zk_expr)
    end
end

function concrete_poly_flux_basis(poly, basis::RefPointData{NDIMS}) where {NDIMS}
    if NDIMS == 1
        # @polyvar x # diff wrt existing vars, new polyvar doesn't work
        poly_x = differentiate.(poly, poly[end].vars[1])
        f = StaticPolynomials.Polynomial.(poly)
        f_x = StaticPolynomials.Polynomial.(poly_x)
        poly_expr = PolynomialSystem(f...)
        poly_x_expr = PolynomialSystem(f_x...)
        return (; poly_expr, poly_x_expr)
    elseif NDIMS == 2
        # @polyvar x y
        poly_x = differentiate.(poly, poly[end].vars[1])
        poly_y = differentiate.(poly, poly[end].vars[2])
        f = StaticPolynomials.Polynomial.(poly)
        f_x = StaticPolynomials.Polynomial.(poly_x)
        f_y = StaticPolynomials.Polynomial.(poly_y)
        poly_expr = PolynomialSystem(f...)
        poly_x_expr = PolynomialSystem(f_x...)
        poly_y_expr = PolynomialSystem(f_y...)
        return (; poly_expr, poly_x_expr, poly_y_expr)
    elseif NDIMS == 3
        # @polyvar x y z
        poly_x = differentiate.(poly, poly[end].vars[1])
        poly_y = differentiate.(poly, poly[end].vars[2])
        poly_z = differentiate.(poly, poly[end].vars[3])
        f = StaticPolynomials.Polynomial.(poly)
        f_x = StaticPolynomials.Polynomial.(poly_x)
        f_y = StaticPolynomials.Polynomial.(poly_y)
        f_z = StaticPolynomials.Polynomial.(poly_z)
        poly_expr = PolynomialSystem(f...)
        poly_x_expr = PolynomialSystem(f_x...)
        poly_y_expr = PolynomialSystem(f_y...)
        poly_z_expr = PolynomialSystem(f_z...)
        return (; poly_expr, poly_x_expr, poly_y_expr, poly_z_expr)
    end
end

# Specialized Basis generation for k-th derivative order
function concrete_poly_flux_basis(poly, basis::RefPointData{NDIMS}, k::Int) where {NDIMS}
    if NDIMS == 1
        # @polyvar x # diff wrt existing vars, new polyvar doesn't work
        poly_xk = deepcopy(poly)
        for i in 1:k # Differentiate k-times
            poly_xk = differentiate.(poly_xk, poly[end].vars[1])
        end
        f = StaticPolynomials.Polynomial.(poly)
        f_xk = StaticPolynomials.Polynomial.(poly_xk)
        poly_expr = PolynomialSystem(f...)
        poly_xk_expr = PolynomialSystem(f_xk...)
        return (; poly_expr,
                poly_x_expr = poly_xk_expr)
    elseif NDIMS == 2
        # @polyvar x y
        poly_xk = deepcopy(poly)
        poly_yk = deepcopy(poly)
        for i in 1:k # Differentiate k-times
            poly_xk = differentiate.(poly_xk, poly[end].vars[1])
            poly_yk = differentiate.(poly_yk, poly[end].vars[2])
        end
        f = StaticPolynomials.Polynomial.(poly)
        f_xk = StaticPolynomials.Polynomial.(poly_xk)
        f_yk = StaticPolynomials.Polynomial.(poly_yk)
        poly_expr = PolynomialSystem(f...)
        poly_xk_expr = PolynomialSystem(f_xk...)
        poly_yk_expr = PolynomialSystem(f_yk...)
        return (; poly_expr,
                poly_x_expr = poly_xk_expr,
                poly_y_expr = poly_yk_expr)
    elseif NDIMS == 3
        # @polyvar x y z
        poly_xk = deepcopy(poly)
        poly_yk = deepcopy(poly)
        poly_zk = deepcopy(poly)
        for i in 1:k # Differentiate k-times
            poly_xk = differentiate.(poly_xk, poly[end].vars[1])
            poly_yk = differentiate.(poly_yk, poly[end].vars[2])
            poly_zk = differentiate.(poly_zk, poly[end].vars[3])
        end
        f = StaticPolynomials.Polynomial.(poly)
        f_xk = StaticPolynomials.Polynomial.(poly_xk)
        f_yk = StaticPolynomials.Polynomial.(poly_yk)
        f_zk = StaticPolynomials.Polynomial.(poly_zk)
        poly_expr = PolynomialSystem(f...)
        poly_xk_expr = PolynomialSystem(f_xk...)
        poly_yk_expr = PolynomialSystem(f_yk...)
        poly_zk_expr = PolynomialSystem(f_zk...)
        return (; poly_expr,
                poly_x_expr = poly_xk_expr,
                poly_y_expr = poly_yk_expr,
                poly_xz_expr = poly_zk_expr)
    end
end

function rbf_block(rbf_expr, basis::RefPointData{NDIMS},
                   X::Vector{SVector{NDIMS, T}}) where {NDIMS, T}
    # Generate RBF Matrix for one interpolation point
    #
    # Inputs:   rbf_expr - RBF Function
    #           X - Input Point Set
    #
    # Outputs:  Φ - RBF Matrix Block

    m = lastindex(X)
    D = Array{SVector{NDIMS, T}, 2}(undef, m, m)

    for j in eachindex(X)
        for i in eachindex(X)
            D[i, j] = X[i] - X[j]
        end
    end

    return rbf_expr.(D)
end

function poly_block(poly_func, basis::RefPointData{NDIMS},
                    X::Vector{SVector{NDIMS, T}}) where {NDIMS, T}
    # Generate the polynomial basis block for one
    #  interpolation point
    #
    # Inputs:   F - StaticPolynomial Array
    #           X - Input Point Set
    #
    # Outputs:  P - Monomial Basis Block

    n = length(poly_func)
    m = lastindex(X)

    P = zeros(T, m, n)

    for i in eachindex(X)
        P[i, :] = StaticPolynomials.evaluate(poly_func, X[i])
    end

    return P
end

function shift_stencil(X_local::Vector{SVector{NDIMS, T}}) where {NDIMS, T}
    # Shift to origin
    # Determine size for Distance Block
    X_shift = similar(X_local) # Rework to use SVector
    X_ = similar(X_local) # Rework to use SVector
    # Assuming first value is the current interpolation point
    x_shift = X_local[1]

    # Shift the stencil nodes so that the center node is the origin
    for j in eachindex(X_local)
        X_shift[j] = X_local[j] - x_shift
    end

    # Add small offset to prevent div-by-0
    X_shift[1] = SVector{NDIMS, T}(tuple((eps(T) for _ in 1:NDIMS)...))

    return X_shift
end

function interpolation_block(R::Matrix, P::Matrix)
    Symmetric(hvcat((2, 2), R, P, P', zeros(size(P)[2], size(P)[2])))
end

function poly_linearoperator(X::SVector{1, T}, poly_func::NamedTuple) where {T}
    # Generate RHS Corresponding to Linear Operators on RBF System
    @unpack poly_expr, poly_x_expr = poly_func
    r_F = StaticPolynomials.evaluate(poly_expr, X)
    r_Fx = StaticPolynomials.evaluate(poly_x_expr, X)

    return (; r_F, r_Fx)
end

function poly_linearoperator(X::SVector{2, T}, poly_func::NamedTuple) where {T}
    # Generate RHS Corresponding to Linear Operators on RBF System
    @unpack poly_expr, poly_x_expr, poly_y_expr = poly_func
    r_F = StaticPolynomials.evaluate(poly_expr, X)
    r_Fx = StaticPolynomials.evaluate(poly_x_expr, X)
    r_Fy = StaticPolynomials.evaluate(poly_y_expr, X)

    return (; r_F, r_Fx, r_Fy)
end

function poly_linearoperator(X::SVector{3, T}, poly_func::NamedTuple) where {T}
    # Generate RHS Corresponding to Linear Operators on RBF System
    @unpack poly_expr, poly_x_expr, poly_y_expr, poly_z_expr = poly_func
    r_F = StaticPolynomials.evaluate(poly_expr, X)
    r_Fx = StaticPolynomials.evaluate(poly_x_expr, X)
    r_Fy = StaticPolynomials.evaluate(poly_y_expr, X)
    r_Fz = StaticPolynomials.evaluate(poly_z_expr, X)

    return (; r_F, r_Fx, r_Fy, r_Fz)
end

# function poly_linearoperator(X::Vector{SVector{3, T}}, poly_func::NamedTuple) where {T}
#     # Generate RHS Corresponding to Linear Operators on RBF System
#     # Vectorized version
#     @unpack poly_expr, poly_x_expr, poly_y_expr, poly_z_expr = poly_func
#     n_p = length(poly_expr)
#     m = lastindex(X)
#     r_F = Matrix{T}(undef, m, n_p)
#     r_Fx = Matrix{T}(undef, m, n_p)
#     r_Fy = Matrix{T}(undef, m, n_p)
#     r_Fz = Matrix{T}(undef, m, n_p)

#     ### Generate Poly righthand side
#     for i in eachindex(X)
#         r_F[i, :] = StaticPolynomials.evaluate(poly_expr, X[i])
#         r_Fx[i, :] = StaticPolynomials.evaluate(poly_x_expr, X[i])
#         r_Fy[i, :] = StaticPolynomials.evaluate(poly_y_expr, X[i])
#         r_Fz[i, :] = StaticPolynomials.evaluate(poly_z_expr, X[i])
#     end

#     return (; r_F, r_Fx, r_Fy, r_Fz)
# end

function rbf_linearoperator(X::Vector{SVector{1, T}}, rbf_func::NamedTuple) where {T}
    # Generate RHS Corresponding to Linear Operators on RBF System
    @unpack rbf_expr, rbf_x_expr = rbf_func
    # n_p = length(F)
    # m = lastindex(X)

    r_F = rbf_expr.(X)
    # 1st derivatives 
    #bx = p .* dx_stencil .* r_stencil.^(p-2)
    r_Fx = rbf_x_expr.(X)

    return (; r_F, r_Fx)
end

function rbf_linearoperator(X::Vector{SVector{2, T}}, rbf_func::NamedTuple) where {T}
    # Generate RHS Corresponding to Linear Operators on RBF System
    @unpack rbf_expr, rbf_x_expr, rbf_y_expr = rbf_func
    # n_p = length(F)
    # m = lastindex(X)

    r_F = rbf_expr.(X)
    # 1st derivatives 
    #bx = p .* dx_stencil .* r_stencil.^(p-2)
    r_Fx = rbf_x_expr.(X)
    r_Fy = rbf_y_expr.(X)

    return (; r_F, r_Fx, r_Fy)
end

function rbf_linearoperator(X::Vector{SVector{3, T}}, rbf_func::NamedTuple) where {T}
    # Generate RHS Corresponding to Linear Operators on RBF System
    @unpack rbf_expr, rbf_x_expr, rbf_y_expr, rbf_z_expr = rbf_func
    # n_p = length(F)
    # m = lastindex(X)

    r_F = rbf_expr.(X)
    # 1st derivatives 
    #bx = p .* dx_stencil .* r_stencil.^(p-2)
    r_Fx = rbf_x_expr.(X)
    r_Fy = rbf_y_expr.(X)
    r_Fz = rbf_z_expr.(X)

    return (; r_F, r_Fx, r_Fy, r_Fz)
end

function assemble_rhs(rbf_rhs::NamedTuple, poly_rhs::NamedTuple, basis::RefPointData{1})
    return [rbf_rhs.r_Fx rbf_rhs.r_F;
            poly_rhs.r_Fx poly_rhs.r_F]
end

function assemble_rhs(rbf_rhs::NamedTuple, poly_rhs::NamedTuple, basis::RefPointData{2})
    return [rbf_rhs.r_Fx rbf_rhs.r_Fy rbf_rhs.r_F;
            poly_rhs.r_Fx poly_rhs.r_Fy poly_rhs.r_F]
end

function assemble_rhs(rbf_rhs::NamedTuple, poly_rhs::NamedTuple, basis::RefPointData{3})
    return [rbf_rhs.r_Fx rbf_rhs.r_Fy rbf_rhs.r_Fz rbf_rhs.r_F;
            poly_rhs.r_Fx poly_rhs.r_Fy poly_rhs.r_Fz poly_rhs.r_F]
end

# Port of generator_operator from RBFD to generate Dx and Dy flux operators
function compute_flux_operator(solver::RBFSolver,
                               domain::PointCloudDomain{2})
    @unpack basis = solver
    @unpack rbf, poly = basis.f
    @unpack points, neighbors, num_points, num_neighbors = domain.pd

    rbf_func = concrete_rbf_flux_basis(rbf, basis)
    poly_func = concrete_poly_flux_basis(poly, basis)

    # Solve RBF interpolation system for all points
    E_loc = zeros(num_points, num_neighbors)
    Dx_loc = zeros(num_points, num_neighbors)
    Dy_loc = zeros(num_points, num_neighbors)
    for e in eachelement(domain, solver)
        # Create Interpolation System
        neighbor_idx = neighbors[e]
        local_points = points[neighbor_idx]
        local_points_shifted = shift_stencil(local_points)
        R = rbf_block(rbf_func.rbf_expr, basis, local_points_shifted)
        P = poly_block(poly_func.poly_expr, basis, local_points_shifted)
        M = interpolation_block(R, P)
        # Assemble RHS
        poly_rhs = poly_linearoperator(local_points_shifted[1], poly_func)
        rbf_rhs = rbf_linearoperator(local_points_shifted, rbf_func)
        rhs = assemble_rhs(rbf_rhs, poly_rhs, basis)
        weights = M \ rhs
        # Extract RBF Stencil Weights
        Dx_loc[e, :] = weights[1:(num_neighbors), 1]
        Dy_loc[e, :] = weights[1:(num_neighbors), 2]
        E_loc[e, :] = weights[1:(num_neighbors), 3]
    end
    # Generate Sparse Matrices from Local Operator Matrices
    idx_rows = repeat((eachindex(points))', num_neighbors)'
    idx_columns = Array{eltype(neighbors[1])}(undef, num_points, num_neighbors) # Change to eltype of existing indices
    for i in eachindex(points)
        idx_columns[i, :] = neighbors[i]
    end
    ### Convert to Generate Sparse Matrix 
    E = sparse(vec(idx_rows), vec(idx_columns), vec(E_loc))
    Dx = sparse(vec(idx_rows), vec(idx_columns), vec(Dx_loc))
    Dy = sparse(vec(idx_rows), vec(idx_columns), vec(Dy_loc))
    return [Dx, Dy]
end

function compute_flux_operator(solver::RBFSolver,
                               domain::PointCloudDomain{1})
    @unpack basis = solver
    @unpack rbf, poly = basis.f
    @unpack points, neighbors, num_points, num_neighbors = domain.pd

    rbf_func = concrete_rbf_flux_basis(rbf, basis)
    poly_func = concrete_poly_flux_basis(poly, basis)

    # Solve RBF interpolation system for all points
    E_loc = zeros(num_points, num_neighbors)
    Dx_loc = zeros(num_points, num_neighbors)
    # Dy_loc = zeros(num_points, num_neighbors)
    for e in eachelement(domain, solver)
        # Create Interpolation System
        neighbor_idx = neighbors[e]
        local_points = points[neighbor_idx]
        local_points_shifted = shift_stencil(local_points)
        R = rbf_block(rbf_func.rbf_expr, basis, local_points_shifted)
        P = poly_block(poly_func.poly_expr, basis, local_points_shifted)
        M = interpolation_block(R, P)
        # Assemble RHS
        poly_rhs = poly_linearoperator(local_points_shifted[1], poly_func)
        rbf_rhs = rbf_linearoperator(local_points_shifted, rbf_func)
        rhs = assemble_rhs(rbf_rhs, poly_rhs, basis)
        weights = M \ rhs
        # Extract RBF Stencil Weights
        Dx_loc[e, :] = weights[1:(num_neighbors), 1]
        # Dy_loc[i, :] = weights[1:(num_neighbors), 2]
        E_loc[e, :] = weights[1:(num_neighbors), 2]
    end
    # Generate Sparse Matrices from Local Operator Matrices
    idx_rows = repeat((eachindex(points))', num_neighbors)'
    idx_columns = Array{eltype(neighbors[1])}(undef, num_points, num_neighbors) # Change to eltype of existing indices
    for i in eachindex(points)
        idx_columns[i, :] = neighbors[i]
    end
    ### Convert to Generate Sparse Matrix 
    E = sparse(vec(idx_rows), vec(idx_columns), vec(E_loc))
    Dx = sparse(vec(idx_rows), vec(idx_columns), vec(Dx_loc))
    # Dy = sparse(vec(idx_rows), vec(idx_columns), vec(Dy_loc))
    return [Dx]
end

function compute_flux_operator(solver::RBFSolver,
                               domain::PointCloudDomain{3})
    @unpack basis = solver
    @unpack rbf, poly = basis.f
    @unpack points, neighbors, num_points, num_neighbors = domain.pd

    rbf_func = concrete_rbf_flux_basis(rbf, basis)
    poly_func = concrete_poly_flux_basis(poly, basis)

    # Solve RBF interpolation system for all points
    E_loc = zeros(num_points, num_neighbors)
    Dx_loc = zeros(num_points, num_neighbors)
    Dy_loc = zeros(num_points, num_neighbors)
    Dz_loc = zeros(num_points, num_neighbors)
    for e in eachelement(domain, solver)
        # Create Interpolation System
        neighbor_idx = neighbors[e]
        local_points = points[neighbor_idx]
        local_points_shifted = shift_stencil(local_points)
        R = rbf_block(rbf_func.rbf_expr, basis, local_points_shifted)
        P = poly_block(poly_func.poly_expr, basis, local_points_shifted)
        M = interpolation_block(R, P)
        # Assemble RHS
        poly_rhs = poly_linearoperator(local_points_shifted[1], poly_func)
        rbf_rhs = rbf_linearoperator(local_points_shifted, rbf_func)
        rhs = assemble_rhs(rbf_rhs, poly_rhs, basis)
        weights = M \ rhs
        # Extract RBF Stencil Weights
        Dx_loc[e, :] = weights[1:(num_neighbors), 1]
        Dy_loc[e, :] = weights[1:(num_neighbors), 2]
        Dz_loc[e, :] = weights[1:(num_neighbors), 3]
        E_loc[e, :] = weights[1:(num_neighbors), 4]
    end
    # Generate Sparse Matrices from Local Operator Matrices
    idx_rows = repeat((eachindex(points))', num_neighbors)'
    idx_columns = Array{eltype(neighbors[1])}(undef, num_points, num_neighbors) # Change to eltype of existing indices
    for i in eachindex(points)
        idx_columns[i, :] = neighbors[i]
    end
    ### Convert to Generate Sparse Matrix 
    E = sparse(vec(idx_rows), vec(idx_columns), vec(E_loc))
    Dx = sparse(vec(idx_rows), vec(idx_columns), vec(Dx_loc))
    Dy = sparse(vec(idx_rows), vec(idx_columns), vec(Dy_loc))
    Dz = sparse(vec(idx_rows), vec(idx_columns), vec(Dz_loc))
    return [Dx, Dy, Dz]
end

# Specialized operator generation for k-th derivative order
function compute_flux_operator(solver::RBFSolver,
                               domain::PointCloudDomain{2}, k::Int)
    # Compute specific derivative operators
    @unpack basis = solver
    @unpack rbf, poly = basis.f
    @unpack points, neighbors, num_points, num_neighbors = domain.pd

    rbf_func = concrete_rbf_flux_basis(rbf, basis, k)
    poly_func = concrete_poly_flux_basis(poly, basis, k)

    # Solve RBF interpolation system for all points
    E_loc = zeros(num_points, num_neighbors)
    Dxk_loc = zeros(num_points, num_neighbors)
    Dyk_loc = zeros(num_points, num_neighbors)
    for e in eachelement(domain, solver)
        # Create Interpolation System
        neighbor_idx = neighbors[e]
        local_points = points[neighbor_idx]
        local_points_shifted = shift_stencil(local_points)
        R = rbf_block(rbf_func.rbf_expr, basis, local_points_shifted)
        P = poly_block(poly_func.poly_expr, basis, local_points_shifted)
        M = interpolation_block(R, P)
        # Assemble RHS
        poly_rhs = poly_linearoperator(local_points_shifted[1], poly_func)
        rbf_rhs = rbf_linearoperator(local_points_shifted, rbf_func)
        rhs = assemble_rhs(rbf_rhs, poly_rhs, basis)
        weights = M \ rhs
        # Extract RBF Stencil Weights
        Dxk_loc[e, :] = weights[1:(num_neighbors), 1]
        Dyk_loc[e, :] = weights[1:(num_neighbors), 2]
        E_loc[e, :] = weights[1:(num_neighbors), 3]
    end
    # Generate Sparse Matrices from Local Operator Matrices
    idx_rows = repeat((eachindex(points))', num_neighbors)'
    idx_columns = Array{eltype(neighbors[1])}(undef, num_points, num_neighbors) # Change to eltype of existing indices
    for i in eachindex(points)
        idx_columns[i, :] = neighbors[i]
    end
    ### Convert to Generate Sparse Matrix 
    E = sparse(vec(idx_rows), vec(idx_columns), vec(E_loc))
    Dxk = sparse(vec(idx_rows), vec(idx_columns), vec(Dxk_loc))
    Dyk = sparse(vec(idx_rows), vec(idx_columns), vec(Dyk_loc))
    return [Dxk, Dyk]
end

function compute_flux_operator(solver::RBFSolver,
                               domain::PointCloudDomain{3}, k::Int)
    # Compute specific derivative operators
    @unpack basis = solver
    @unpack rbf, poly = basis.f
    @unpack points, neighbors, num_points, num_neighbors = domain.pd

    rbf_func = concrete_rbf_flux_basis(rbf, basis, k)
    poly_func = concrete_poly_flux_basis(poly, basis, k)

    # Solve RBF interpolation system for all points
    E_loc = zeros(num_points, num_neighbors)
    Dxk_loc = zeros(num_points, num_neighbors)
    Dyk_loc = zeros(num_points, num_neighbors)
    Dzk_loc = zeros(num_points, num_neighbors)
    for e in eachelement(domain, solver)
        # Create Interpolation System
        neighbor_idx = neighbors[e]
        local_points = points[neighbor_idx]
        local_points_shifted = shift_stencil(local_points)
        R = rbf_block(rbf_func.rbf_expr, basis, local_points_shifted)
        P = poly_block(poly_func.poly_expr, basis, local_points_shifted)
        M = interpolation_block(R, P)
        # Assemble RHS
        poly_rhs = poly_linearoperator(local_points_shifted[1], poly_func)
        rbf_rhs = rbf_linearoperator(local_points_shifted, rbf_func)
        rhs = assemble_rhs(rbf_rhs, poly_rhs, basis)
        weights = M \ rhs
        # Extract RBF Stencil Weights
        Dxk_loc[e, :] = weights[1:(num_neighbors), 1]
        Dyk_loc[e, :] = weights[1:(num_neighbors), 2]
        Dzk_loc[e, :] = weights[1:(num_neighbors), 3]
        E_loc[e, :] = weights[1:(num_neighbors), 4]
    end
    # Generate Sparse Matrices from Local Operator Matrices
    idx_rows = repeat((eachindex(points))', num_neighbors)'
    idx_columns = Array{eltype(neighbors[1])}(undef, num_points, num_neighbors) # Change to eltype of existing indices
    for i in eachindex(points)
        idx_columns[i, :] = neighbors[i]
    end
    ### Convert to Generate Sparse Matrix 
    E = sparse(vec(idx_rows), vec(idx_columns), vec(E_loc))
    Dxk = sparse(vec(idx_rows), vec(idx_columns), vec(Dxk_loc))
    Dyk = sparse(vec(idx_rows), vec(idx_columns), vec(Dyk_loc))
    Dzk = sparse(vec(idx_rows), vec(idx_columns), vec(Dzk_loc))
    return [Dxk, Dyk, Dzk]
end

function compute_flux_operator(solver::RBFSolver,
                               domain::PointCloudDomain{1}, k::Int)
    # Compute specific derivative operators
    @unpack basis = solver
    @unpack rbf, poly = basis.f
    @unpack points, neighbors, num_points, num_neighbors = domain.pd

    rbf_func = concrete_rbf_flux_basis(rbf, basis, k)
    poly_func = concrete_poly_flux_basis(poly, basis, k)

    # Solve RBF interpolation system for all points
    E_loc = zeros(num_points, num_neighbors)
    Dxk_loc = zeros(num_points, num_neighbors)
    for e in eachelement(domain, solver)
        # Create Interpolation System
        neighbor_idx = neighbors[e]
        local_points = points[neighbor_idx]
        local_points_shifted = shift_stencil(local_points)
        R = rbf_block(rbf_func.rbf_expr, basis, local_points_shifted)
        P = poly_block(poly_func.poly_expr, basis, local_points_shifted)
        M = interpolation_block(R, P)
        # Assemble RHS
        poly_rhs = poly_linearoperator(local_points_shifted[1], poly_func)
        rbf_rhs = rbf_linearoperator(local_points_shifted, rbf_func)
        rhs = assemble_rhs(rbf_rhs, poly_rhs, basis)
        weights = M \ rhs
        # Extract RBF Stencil Weights
        Dxk_loc[e, :] = weights[1:(num_neighbors), 1]
        E_loc[e, :] = weights[1:(num_neighbors), 2]
    end
    # Generate Sparse Matrices from Local Operator Matrices
    idx_rows = repeat((eachindex(points))', num_neighbors)'
    idx_columns = Array{eltype(neighbors[1])}(undef, num_points, num_neighbors) # Change to eltype of existing indices
    for i in eachindex(points)
        idx_columns[i, :] = neighbors[i]
    end
    ### Convert to Generate Sparse Matrix 
    E = sparse(vec(idx_rows), vec(idx_columns), vec(E_loc))
    Dxk = sparse(vec(idx_rows), vec(idx_columns), vec(Dxk_loc))
    return [Dxk]
end
