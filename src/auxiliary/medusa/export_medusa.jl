# Function for exporting Medusa point datasets to VTK
# Exports as VTKCellTypes.VTK_VERTEX

### Needs to be reworked to whatever new structure is implemented

using WriteVTK

# export_soln(name, dir, meshname, markernames, sol, endtime, frames)
function export_soln_medusa(name, dir, sol, positions, ϵ_s, ϵ_rv_s, ϵ_uw_s, ϵ_c_s)
    npoints = length(positions)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]
    for i in 0:frames
        curr_frame = string(i)
        output_name = dir * "/" * name * "_" * curr_frame

        sol_ = sol(timestep * i)
        # momentum_x = [[sol_[2,x], 0] for x in eachindex(sol_[2,:])]
        # momentum_y = [[sol_[3,x], 0] for x in eachindex(sol_[3,:])]
        momentum_x = hcat(sol_[:, 2], zeros(length(sol_[:, 2])))
        momentum_y = hcat(zeros(length(sol_[:, 3])), sol_[:, 3])
        # momentum = hcat(sol[2,:],sol[3,:])

        # Extract additional saved values 
        # eps_rv = saved_values.saveval[i+1]
        ϵ = ϵ_s[i + 1]
        ϵ_rv = ϵ_rv_s[i + 1]
        ϵ_uw = ϵ_uw_s[i + 1]
        ϵ_c = ϵ_c_s[i + 1]

        # Separate subvariable
        ρ = sol_[:, 1]
        mx = sol_[:, 2]
        my = sol_[:, 3]
        mz = my .* 0.0
        E_int = sol_[:, 4]
        vx = mx ./ ρ
        vy = my ./ ρ
        pr = (γ - 1) .* (E_int .- ρ .* (vx .^ 2 .+ vy .^ 2) ./ 2)
        T = pr ./ ρ
        for i in eachindex(T)
            if T[i] < 0
                T[i] = 0.0
            end
        end
        # ϵ_uw = (1 / 2) .* h_loc .* (sqrt.(vx .^ 2 .+ vy .^ 2) .+ sqrt.(γ .* T))
        # Number of total points 
        vtk_grid(output_name, x, y, z, cells) do vtk
            vtk["density", VTKPointData()] = ρ
            vtk["momentum_x", VTKPointData()] = mx
            vtk["momentum_y", VTKPointData()] = my
            vtk["momentum_z", VTKPointData()] = mz
            vtk["momentum", VTKPointData()] = (mx, my)
            vtk["vel_x", VTKPointData()] = vx
            vtk["vel_y", VTKPointData()] = vy
            vtk["vel_z", VTKPointData()] = mz
            vtk["vel", VTKPointData()] = (vx, vy)
            #vtk["momentum"] = momentum
            vtk["energy", VTKPointData()] = E_int
            vtk["pressure", VTKPointData()] = pr
            vtk["temperature", VTKPointData()] = T
            vtk["upwind_visc", VTKPointData()] = ϵ_uw
            vtk["residual_visc", VTKPointData()] = ϵ_rv
            vtk["epsilon", VTKPointData()] = ϵ
            vtk["epsilon_c", VTKPointData()] = ϵ_c
        end
    end
end