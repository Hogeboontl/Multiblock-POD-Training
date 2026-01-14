
import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI

# -------------------------
# Thermal conductivity kappa
# -------------------------
class KappaExpression:
    def __init__(self, k_0, k_1, h, thick_Sio2, tol):
        self.k_0 = k_0
        self.k_1 = k_1
        self.h = h
        self.thick_Sio2 = thick_Sio2
        self.tol = tol

    def eval(self, x):
        threshold = self.h - self.thick_Sio2
        vals = np.where(x[:, 2] <= threshold + self.tol, self.k_0, self.k_1)
        return vals

# -------------------------
# Density
# -------------------------
class DensityExpression:
    def __init__(self, rho_silicon, rho_oxide, h, thick_Sio2, tol):
        self.rho_silicon = rho_silicon
        self.rho_oxide = rho_oxide
        self.h = h
        self.thick_Sio2 = thick_Sio2
        self.tol = tol

    def eval(self, x):
        threshold = self.h - self.thick_Sio2
        vals = np.where(x[:, 2] <= threshold + self.tol, self.rho_silicon, self.rho_oxide)
        return vals

# -------------------------
# Specific heat
# -------------------------
class SpecificHeatExpression:
    def __init__(self, c_silicon, c_oxide, h, thick_Sio2, tol):
        self.c_silicon = c_silicon
        self.c_oxide = c_oxide
        self.h = h
        self.thick_Sio2 = thick_Sio2
        self.tol = tol

    def eval(self, x):
        threshold = self.h - self.thick_Sio2
        vals = np.where(x[:, 2] <= threshold + self.tol, self.c_silicon, self.c_oxide)
        return vals

# -------------------------
# Extended source / power density
# -------------------------
class ExtendedSourceExpression:
    
    def __init__(self, flp, pd_row, center_index, mesh_tags, tag_volumes,
                 chip_thickness, active_thickness, power_max, tol, pg_change,step, seed=None):
        self.flp = flp
        self.pd_row = pd_row
        self.center_index = center_index
        self.mesh_tags = mesh_tags  
        self.tag_volumes = tag_volumes
        self.chip_thickness = chip_thickness
        self.active_thickness = active_thickness
        self.power_max = power_max
        self.tol = tol
        self.rng = np.random.default_rng(seed if seed is not None else center_index)
        self.power_map = self._generate_power_map()
        self.pg_change=pg_change

    def update_power(self, step):
        if step % self.pg_change == 0:
            self.power_map = self._generate_power_map()

    def _generate_power_map(self):
        power_map = {}
        for region_id in range(1, 10):  # regions 1..9
            if region_id == 5:  # center block
                power_map[region_id] = self.pd_row[self.center_index]
            else:
                volume = self.tag_volumes.get(region_id, 1.0)
                power_map[region_id] = self.rng.uniform(0.0, self.power_max) / volume
        return power_map

    def to_function(self, V):
        """Return a FEniCSx DG0 Function representing the power density in the active layer."""
        mesh = V.mesh
        tdim = mesh.topology.dim
        cell_indices = np.arange(mesh.topology.index_map(tdim).size_local)
        
        # Compute cell centers
        x = mesh.geometry.x
        conn = mesh.topology.connectivity(tdim, 0).array.reshape((-1, tdim + 1))
        cell_centers = x[conn].mean(axis=1)

        # Allocate array for function values
        vals = np.zeros(len(cell_centers), dtype=np.float64)

        # Active layer bounds in z
        lower = self.chip_thickness - 2 * self.active_thickness - self.tol
        upper = self.chip_thickness - self.active_thickness + self.tol  # add tol to upper bound

        # Assign power only to cells inside active layer
        for i, cell in enumerate(cell_indices):
            region_id = self.mesh_tags.values[cell]
            z = cell_centers[i, 2]

            if lower <= z <= upper:
                vals[i] = self.power_map.get(region_id, 0.0)
            else:
                vals[i] = 0.0  # explicitly zero outside active layer

        # Create DG0 function and assign per-cell values
        Vdg = fem.FunctionSpace(mesh, ("DG", 0))
        f = fem.Function(Vdg)
        f.x.array[:] = vals
        f.x.scatter_forward()

        return f
