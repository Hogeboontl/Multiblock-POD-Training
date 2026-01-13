import numpy as np
import torch
import h5py
import os
from utils.get_C_matrix import *
from utils.get_pod_modes import *
from utils.get_P_matrix import *
from utils.get_G_matrix import *
from utils.get_G_interface import *
from utils.ODE_solver import *
from config import parser, config_args

# Load floorplan & args 
flp = np.loadtxt('floorplan_AMD.txt')
if flp.ndim == 1:
    flp = flp[np.newaxis, :] 
num_blocks = len(flp)
args = parser.parse_args()
device = torch.device("cpu" if args.cuda == -1 else f"cuda:{args.cuda}")
dt = args.sampling_interval
file_path = args.file_path
save_dir = os.path.join(file_path, "POD_solutions")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "pod_solution.h5")

# Compute all components 
pod_modes, _ = get_pod_modes()  # ensures POD modes are saved for C computation
C_list = calc_C()       # list of per-block C matrices
P_list = calc_P()       # list of per-block P matrices [num_modes_i x time_steps]
interfaces = get_block_interfaces()
compute_POD_gradients(interfaces)
G_list = calc_G()       # list of diagonal G matrices
if num_blocks != 1:
    Gb_list = get_G_interfaces(interfaces)   # interface G terms
    Gp_list = get_G_mode_coupling(interfaces)  # coupling terms


num_modes_per_block = G_list[0].shape[0]
total_modes = num_modes_per_block * num_blocks

if num_blocks == 1:
    True_G_tensor = G_list[0].detach().clone().to(device).double()
else:
    # Build block matrix G 
    block_rows = []
    for i in range(num_blocks):
        row_blocks = []
        for j in range(num_blocks):
            if j == i:
                block = (G_list[i] + sum(Gb_list[i])).detach().clone().to(device)
            else:
                if Gp_list[i][j]:
                    block = torch.tensor(np.sum(Gp_list[i][j], axis=0), device=device)
                else:
                    block = torch.zeros(num_modes_per_block, num_modes_per_block, device=device)
            row_blocks.append(block)
        block_rows.append(torch.cat(row_blocks, dim=1))

    True_G_tensor = torch.cat(block_rows, dim=0).double() # [total_modes, total_modes]

# Flatten C into block-diagonal (keep original C matrices)
C_tensor = torch.block_diag(*[C.to(device).double() for C in C_list]) # [total_modes, total_modes]

# Flatten P into vertical stack 
P_tensor = torch.cat(P_list, dim=0).to(device).double()  # [total_modes, time_steps]

print("\n=== Matrix Diagnostics ===")
print(f"C_tensor shape: {C_tensor.shape}")
print(f"C_tensor range: [{C_tensor.min():.3e}, {C_tensor.max():.3e}]")
print(f"C_tensor diagonal (first 5): {torch.diag(C_tensor)[:5]}")

print(f"\nTrue_G_tensor shape: {True_G_tensor.shape}")
print(f"True_G_tensor range: [{True_G_tensor.min():.3e}, {True_G_tensor.max():.3e}]")
print(f"True_G_tensor diagonal (first 5): {torch.diag(True_G_tensor)[:5]}")

print(f"\nP_tensor shape: {P_tensor.shape}")
print(f"P_tensor range: [{P_tensor.min():.3e}, {P_tensor.max():.3e}]")

# Check matrix scales
C_scale = C_tensor.abs().median().item()
G_scale = True_G_tensor.abs().median().item()
print(f"\nC median magnitude: {C_scale:.3e}")
print(f"G median magnitude: {G_scale:.3e}")
#print(f"G/C ratio: {G_scale/C_scale:.3e}")


# Solve ODE 
solver = POD_ODE_Solver(
    C=C_tensor,
    G=True_G_tensor,
    P=P_tensor,
    time_steps=args.num_steps,
    num_modes=total_modes, 
    dt=dt,
    sampling_interval=args.sampling_interval,
    multiple=False
)

solution = solver.solve()
solution_np = solution.cpu().numpy()

#print(solution)

mode_indices = []
start = 0
for i in range(num_blocks):
    end = start + num_modes_per_block
    mode_indices.append((start, end))
    start = end

#Reconstruct full temperature per block and save
for t in range(solution_np.shape[0]):  # time steps
    for b, (start, end) in enumerate(mode_indices):
        # slice mode coefficients for this block
        block_coeffs = solution_np[t, start:end]  # [num_modes_block]
        #print(block_coeffs)

        # reconstruct full-order temperature
        full_temp = pod_modes[b] @ block_coeffs  # [num_nodes_block]
        #print(full_temp)

        # save to HDF5
        block_file = os.path.join(save_dir, f"time_{t:04d}_block_{b}.h5")
        with h5py.File(block_file, "w") as hf:
            hf.create_dataset(f"block_{b}_temperature", data=full_temp)

print(f"Full POD temperature solutions saved to {save_dir}")
