#G matrix will use central difference due to uniformity of mesh, may have to be reworked to barycentric for non uniform in the future
#for now, central difference is less computationally heavy and almost as accurate if the mesh is fine

import os
import numpy as np
import torch
from dolfinx import fem, io
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace, form,assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from config import parser, config_args
import meshio
import h5py

comm = MPI.COMM_WORLD
# ------------------ compute_POD_gradients (fixed) ------------------
def compute_POD_gradients(block_interfaces):
    args = parser.parse_args()
    file_path = args.file_path
    device = torch.device("cpu" if args.cuda == -1 else f"cuda:{args.cuda}")

    all_pod_modes = np.load(f"{file_path}/POD/POD_MODES.npy", allow_pickle=True)
    flp = np.loadtxt("floorplan_AMD.txt")
    if flp.ndim == 1:
        flp = flp[np.newaxis, :]

    os.makedirs(f"{file_path}/POD_surfaces", exist_ok=True)

    for block_index in range(len(flp)):
        POD_mode = torch.tensor(all_pod_modes[block_index], device=device, dtype=torch.float64)
        num_nodes, num_modes = POD_mode.shape

        # Load mesh
        mesh_file = f"{file_path}/xdmf/block_{block_index}_center.xdmf"
        with XDMFFile(comm, mesh_file, "r") as xdmf:
            msh = xdmf.read_mesh(name="mesh")

        V = FunctionSpace(msh, ("CG", 1))
        W = VectorFunctionSpace(msh, ("CG", 1))  # nodal vector space

        grad_modes = torch.zeros((num_nodes, num_modes, 3), device=device)

        for mode_idx in range(num_modes):
            f = fem.Function(V)
            f.x.array[:] = POD_mode[:, mode_idx].cpu().numpy()

            # L2 projection to nodal CG vector space
            u = ufl.TrialFunction(W)
            v = ufl.TestFunction(W)
            a = ufl.inner(u, v) * ufl.dx
            L = ufl.inner(ufl.grad(f), v) * ufl.dx

            problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                "ksp_type": "cg",
                "pc_type": "jacobi",
                "ksp_rtol": 1e-10
            })
            grad_f = problem.solve()
            grad_modes[:, mode_idx, :] = torch.tensor(grad_f.x.array[:].reshape(-1, 3), device=device)

        np.save(f"{file_path}/POD/block_{block_index}_grad_modes.npy", grad_modes.cpu().numpy())

    print("\nAll gradients computed using nodal L2 projection\n")

# ------------------ calc_G (fixed) ------------------
def calc_G():
    args = parser.parse_args()
    file_path = args.file_path
    device = torch.device("cpu" if args.cuda == -1 else f"cuda:{args.cuda}")

    flp = np.loadtxt("floorplan_AMD.txt")
    if flp.ndim == 1:
        flp = flp[np.newaxis, :]

    G_list = []
    k_silicon = torch.tensor(config_args["training_config"]["k_0"][0], dtype=torch.float32, device=device)

    for i in range(len(flp)):
        w0, h0, x0, y0 = flp[i]

        # Load gradients and nodal kappa
        grad_modes = torch.tensor(np.load(f"{file_path}/POD/block_{i}_grad_modes.npy"),
                                  dtype=torch.float32, device=device)
        kappa = torch.tensor(np.load(f"{file_path}/matrix_necessities/kappa_block{i}.npy"),
                             dtype=torch.float32, device=device)

        # Load mesh
        with io.XDMFFile(MPI.COMM_WORLD, f"{file_path}/xdmf/block_{i}_center.xdmf", "r") as xdmf:
            m = xdmf.read_mesh(name="mesh")

        V = FunctionSpace(m, ("CG", 1))
        num_nodes = V.dofmap.index_map.size_local

        # Node weights (lumped mass)
        cells = m.topology.connectivity(m.topology.dim, 0)
        coords = m.geometry.x
        node_weights = np.zeros(num_nodes)
        for c in range(m.topology.index_map(m.topology.dim).size_local):
            verts = cells.links(c)
            vcoords = coords[verts]
            vol = abs(np.dot(vcoords[1]-vcoords[0], np.cross(vcoords[2]-vcoords[0], vcoords[3]-vcoords[0]))) / 6
            node_weights[verts] += vol / len(verts)
        node_weights_t = torch.tensor(node_weights, device=device, dtype=torch.float32)

        # Assemble G
        _, num_modes, _ = grad_modes.shape
        G = torch.zeros((num_modes, num_modes), device=device)
        for n in range(num_nodes):
            g = grad_modes[n]          # (num_modes,3)
            weight = kappa[n] * node_weights_t[n]
            G += weight * (g @ g.T)

        # Bottom surface contribution
        tol = 1e-8
        bottom_mask = coords[:, 2] <= tol
        bottom_idx = np.nonzero(bottom_mask)[0]
        if len(bottom_idx) > 0:
            grad_z = grad_modes[bottom_idx, :, 2]
            area = (w0*h0)/len(bottom_idx)
            G += k_silicon * area * (grad_z.T @ grad_z)

        G_list.append(G)

    print("All G matrices computed (nodal L2 projection)")
    return G_list







