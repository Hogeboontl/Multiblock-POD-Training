#this file aims to plot eigvals so we can see the number of valid pod modes

import matplotlib.pyplot as plt 
import torch
import numpy as np
import matplotlib
from config import parser, config_args
from utils.get_pod_modes import *


matplotlib.use('Agg')
args = parser.parse_args()
num_modes = args.num_modes

_, eigvals = get_pod_modes()
fig,ax = plt.subplots()
ax.semilogy(np.arange(1,num_modes+1),eigvals[0][:num_modes])
fig.savefig("eigvals.png")

#plt.show()