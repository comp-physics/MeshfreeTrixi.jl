using MeshfreeTrixi
using LinearAlgebra

# Test time history reconstruction 
approx_order = 5
time_history = zeros(approx_order + 1)
sol_history = zeros(approx_order + 1, approx_order + 1)
t = 0.0
u = zeros(approx_order + 1)
approx_du = zeros(approx_order + 1)
time_weights = zeros(approx_order + 1)
success_iter = approx_order + 1

for i in 1:(approx_order + 1)
    global t += 1.0
    global u .+= 1.0
    MeshfreeTrixi.shift_soln_history!(time_history, sol_history, t, u)
end

approx_du .= 0.0
num_time_points = min(success_iter + 1, approx_order + 1)
if success_iter > 0
    # Update the time weights for the current number of time points
    MeshfreeTrixi.time_deriv_weights!(@view(time_weights[1:num_time_points]),
                                      @view(time_history[1:num_time_points]))

    for i in 1:num_time_points
        approx_du .+= time_weights[i] .* sol_history[:, i]
    end
end

@test approx_du ≈ ones(approx_order + 1)