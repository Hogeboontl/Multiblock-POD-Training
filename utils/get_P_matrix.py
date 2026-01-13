import numpy as np
from config import parser, config_args
import torch
from dolfinx import fem, io
from mpi4py import MPI

def calc_P():
    flp = np.loadtxt("floorplan_AMD.txt")
    if flp.ndim == 1:
        flp = flp[np.newaxis, :] 
    
    args = parser.parse_args()
    file_path = args.file_path
    thickness = args.h
    active_thickness = config_args['training_config']['active_thickness'][0]
    device = torch.device("cpu" if args.cuda == -1 else f"cuda:{args.cuda}")
    
    # Load precomputed power density
    pd_np = np.load(f"{file_path}/matrix_necessities/pd.npy")  # shape: (num_steps, n_blocks)
    pd = torch.tensor(pd_np, device=device, dtype=torch.float32)
    
    # Load POD modes
    modes_list = np.load(f"{file_path}/POD/POD_MODES.npy", allow_pickle=True)
    modes_list = [torch.tensor(m, device=device, dtype=torch.float32) for m in modes_list]
    
    P_list = []
    
    for i in range(len(flp)):
        w, h, _, _ = flp[i]
        
        # Load xdmf mesh
        xdmf_path = f'{file_path}/xdmf/block_{i}_center.xdmf'
        with io.XDMFFile(MPI.COMM_WORLD, xdmf_path, "r") as xdmf:
            m = xdmf.read_mesh(name="mesh")

        # Get coordinates and setup topology
        coords = m.geometry.x
        m.topology.create_connectivity(m.topology.dim, 0)
        cells = m.topology.connectivity(m.topology.dim, 0)

        # Compute nodal volumes via mass lumping
        def tet_volume(v0, v1, v2, v3):
            return abs(np.dot((v1-v0), np.cross((v2-v0), (v3-v0)))) / 6.0

        num_nodes = coords.shape[0]
        node_weights = np.zeros(num_nodes)

        for c in range(m.topology.index_map(m.topology.dim).size_local):
            verts = cells.links(c)
            vcoords = coords[verts]
            vol = tet_volume(*vcoords)
            for v in verts:
                node_weights[v] += vol / len(verts)

        # Convert to torch tensors
        coords_torch = torch.tensor(coords, device=device, dtype=torch.float32)
        node_weights_t = torch.tensor(node_weights, device=device, dtype=torch.float32)
        
        # Identify nodes in ACTIVE LAYER (top portion of chip)
        z_max = coords_torch[:, 2].max()
        z_active_bottom = z_max - active_thickness
        tol = 1e-8
        
        active_mask = coords_torch[:, 2] >= (z_active_bottom - tol)
        active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        active_node_weights = node_weights_t[active_indices]
        
        # Get POD modes (in FEniCSx mesh ordering)
        modes = modes_list[i]  # [num_nodes, num_modes]
        num_modes = modes.shape[1]
        num_timesteps = pd.shape[0]
        
        # pd[:, i] is power density (W/m³) in the active layer
        block_power_density = pd[:, i]  # [num_timesteps]
        
        # Initialize P matrix: [num_modes, num_timesteps]
        P = torch.zeros((num_modes, num_timesteps), device=device, dtype=torch.float32)
        
        # Vectorized calculation for efficiency
        phi_active = modes[active_indices, :]
        
        # Weighted modes: [num_active_nodes, num_modes]
        weighted_phi = phi_active * active_node_weights.unsqueeze(1)
        

        # Matrix multiplication: [num_modes, num_active_nodes] @ [num_active_nodes, 1] for each t
        for t in range(num_timesteps):
            q_t = block_power_density[t]
            P[:, t] = q_t * torch.sum(weighted_phi, dim=0)
        
        P_list.append(P)
    
    print("\nAll P matrices computed")
    return P_list