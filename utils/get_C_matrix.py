import numpy as np
from config import parser, config_args
from .expressions import *
import torch
from dolfinx import io, mesh
from dolfinx.io import XDMFFile
from mpi4py import MPI

#torch.float32 will be the standard
#evaluated at each node
#node alignment is the submesh vertex ordering
def calc_C():
    args = parser.parse_args()
    num_modes = args.num_modes
    device = torch.device("cpu" if args.cuda == -1 else f"cuda:{args.cuda}")
    file_path = args.file_path
    modes = np.load(f"{file_path}/POD/POD_MODES.npy", allow_pickle = True) 
    flp = np.loadtxt('floorplan_AMD.txt')# load floorplan will need adjusting for consumer code, as well as loop
    if flp.ndim == 1:
        flp = flp[np.newaxis, :] 
    C_list = []
    height = args.h
    x_divs=args.x_dim
    y_divs=args.y_dim
    z_divs=args.z_dim

    for k in range(len(flp)):  # loop over blocks
        node_weights = np.load(f"{file_path}/matrix_necessities/block_{k}_node_weights.npy")

        # Convert node_weights to torch tensor
        node_weights_t = torch.tensor(node_weights, device=device, dtype=torch.float32)
        
        ds = torch.tensor(np.load(f"{file_path}/matrix_necessities/ds_vals_block{k}.npy"),device=device, dtype = torch.float32)
        u_block = torch.tensor(modes[k],device = device,dtype = torch.float32)
        weighted = (ds[:, None] * u_block) * node_weights_t[:, None]
        C = weighted.T @ u_block
        C_list.append(C) # C matrix gives a value for each POD pair (so Num_modes by num_modes)
    print("all C matrices computed")
    return torch.stack(C_list)