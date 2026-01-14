import torch
import h5py
import numpy as np
from config import parser, config_args
import os
from dolfinx import io, mesh
from dolfinx.io import XDMFFile
from mpi4py import MPI
from dolfinx.fem import FunctionSpace
import meshio


#could benefit from GPU acceleration and MPI parallelization

if __name__ == "__main__": 
    flp = np.loadtxt('floorplan_AMD.txt')
    if flp.ndim == 1:
        flp = flp[np.newaxis, :] 
    args = parser.parse_args()
    time_steps = args.num_steps
    nx, ny, nz = args.x_dim, args.y_dim, args.z_dim
    thickness = args.h
    device = torch.device("cpu" if args.cuda == -1 else f"cuda:{args.cuda}")
    num_modes = args.num_modes
    absfile_path = args.file_path
    POD_modes = []
    all_eigvals = []
    
    for i in range(len(flp)):
        snapshots = []
        for step in range(time_steps):
            file_path = f'{absfile_path}/xdmf_sol/block_{i}_center_step_{step}.h5'
            with h5py.File(file_path,'r') as hf:
                temp = hf["temperature"][:]
                snapshots.append(temp)

        node_weights = np.load(f"{absfile_path}/matrix_necessities/block_{i}_node_weights.npy")

        # Matrix is now row = node and column = snapshot
        X = torch.tensor(np.stack(snapshots, axis=1), dtype=torch.float32, device=device)
        node_weights_t = torch.tensor(node_weights, dtype=torch.float32, device=device)
        # Get the transposed version and get dot product and then divide by num of snapshots

        A = (X.T @ X) / time_steps #adjust for integration
        
        # Solve for eigenvalue
        eigvals, eigvecs = torch.linalg.eigh(A)
    
        # Flip to descending order
        eigvals = eigvals.flip(0) # Just a 1d tensor, not a matrix
        eigvecs = eigvecs.flip(1)

        eigvals = eigvals / eigvals[0]

        # Calculate POD modes
        modes = (X @ eigvecs)
        for j in range(num_modes):
            norm = torch.sqrt(torch.sum(node_weights_t * (modes[:, j]**2)))
            modes[:,j] = modes[:,j] / norm
 

        block_eigvals = eigvals[:num_modes].cpu().numpy()

        block_modes = modes[:, :num_modes].cpu().numpy()
        
        # POD modes is a list of 2d arrays, in which the columns are the POD modes
        # and the row corresponds to the node indice per block
        POD_modes.append(block_modes)
        all_eigvals.append(block_eigvals)


        
        # ----------------  VTU write ----------------
        comm = MPI.COMM_WORLD

        xdmf_file = f"{absfile_path}/xdmf/block_{i}_center.xdmf"
        with XDMFFile(comm, xdmf_file, "r") as infile:
            submesh = infile.read_mesh(name="mesh")

        # Ensure cell-to-vertex connectivity exists
        submesh.topology.create_connectivity(submesh.topology.dim, 0)

        # Extract geometry and topology
        points = submesh.geometry.x
        cells = submesh.topology.connectivity(
            submesh.topology.dim, 0
        ).array.reshape(-1, 4)

        # Sanity check: CG1 => one DOF per vertex
        assert block_modes.shape[0] == points.shape[0], (
            "block_modes must have one value per mesh vertex (CG1 required)"
        )
        # Each POD mode is vertex-based point data
        point_data = {
            f"POD_mode_{k}": block_modes[:, k]
            for k in range(block_modes.shape[1])
        }

        # Write VTU
        os.makedirs(os.path.join(absfile_path, "POD"), exist_ok=True)
        vtu_file = f"{absfile_path}/POD/block_{i}_POD_modes.vtu"

        meshio_mesh = meshio.Mesh(
            points=points,
            cells=[("tetra", cells)],
            point_data=point_data
        )

        meshio.write(vtu_file, meshio_mesh)

        # ---------------------------------------------------
            
    os.makedirs(os.path.join(absfile_path, "POD"), exist_ok=True)
    
    # Save POD modes
    np.save(f"{absfile_path}/POD/POD_MODES.npy", POD_modes, allow_pickle=True)
    
    # Save eigenvalues so gradients can be properly scaled
    np.save(f"{absfile_path}/POD/POD_EIGVALS.npy", all_eigvals, allow_pickle=True)
    
    print("all pod modes calculated")
    print("eigenvalues saved for gradient scaling")







    