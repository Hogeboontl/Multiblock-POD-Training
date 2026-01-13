import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from config import parser

# --- Parse arguments ---
args = parser.parse_args()
file_path = args.file_path

block_id = 0
time_step = 0  # choose the time step to compare

# --- Load precomputed full POD temperature ---
POD_file = os.path.join(file_path, f"POD_solutions/time_{time_step:04d}_block_{block_id}.h5")
with h5py.File(POD_file, "r") as hf:
    POD_temp = hf[f"block_{block_id}_temperature"][:]  # [num_nodes]

print(f"Loaded POD temperature for block {block_id} at time step {time_step}, shape: {POD_temp.shape}")

# --- Load FEM solution for the same block/time ---
FEM_file = os.path.join(file_path, f"xdmf_sol/block_{block_id}_center_step_{time_step}.h5")
with h5py.File(FEM_file, "r") as hf:
    FEM_temp = hf["temperature"][:]  # [num_nodes]

print(f"Loaded FEM temperature for block {block_id} at time step {time_step}, shape: {FEM_temp.shape}")

# --- Sanity check ---
if POD_temp.shape != FEM_temp.shape:
    raise ValueError(f"Shape mismatch: POD {POD_temp.shape} vs FEM {FEM_temp.shape}")

# --- Plot comparison (POD vs FEM) ---
plt.figure(figsize=(12, 6))

num_nodes = len(FEM_temp)
sample_step = 50
indices = np.linspace(0, num_nodes - 1, sample_step, dtype=int)
x_axis = np.arange(sample_step)

plt.scatter(x_axis, FEM_temp[indices], color='blue', label='FEM', s=30, marker='o', alpha=0.7)
plt.scatter(x_axis, POD_temp[indices], color='red', label='POD', s=30, marker='s',
            facecolors='none', edgecolors='red', alpha=0.7)

plt.xlabel("Sampled Node Index")
plt.ylabel("Temperature (K)")
plt.title(f"Block {block_id} – Time step {time_step}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("POD_vs_FEM_comparison.png", dpi=300)
print("Comparison plot saved as POD_vs_FEM_comparison.png")

# --- Compute percent error ---
percent_error = 100.0 * np.abs(POD_temp - FEM_temp) / np.abs(FEM_temp)

# --- Plot percent error ---
plt.figure(figsize=(12, 6))
plt.plot(x_axis, percent_error[indices], color='purple', marker='d', linestyle='--', alpha=0.8)

plt.xlabel("Sampled Node Index")
plt.ylabel("Percent Error (%)")
plt.title(f"Percent Error between POD and FEM – Block {block_id}, Time step {time_step}")
plt.grid(True)
plt.tight_layout()
plt.savefig("POD_vs_FEM_percent_error.png", dpi=300)
print("Percent error plot saved as POD_vs_FEM_percent_error.png")





