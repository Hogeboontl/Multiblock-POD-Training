#this file aims to plot eigvals so we can see the number of valid pod modes

import matplotlib.pyplot as plt 
import torch
import numpy as np
import matplotlib
from config import parser, config_args


matplotlib.use('Agg')
args = parser.parse_args()
num_modes = args.num_modes
file_path = args.file_path
file_path = f'{file_path}/POD/POD_EIGVALS.npy'

eigvals = np.load(file_path)
fig,ax = plt.subplots()
ax.semilogy(np.arange(1,num_modes+1),eigvals[0][:num_modes])
fig.savefig("eigvals.png")

#plt.show()