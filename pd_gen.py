#generates the random powertrace

import numpy as np
from config import parser, config_args
import random

args = parser.parse_args()
flp = np.loadtxt('floorplan_AMD.txt') #load floorplan of each functional unit using numpy
if flp.ndim == 1: #ensures it works for a floorplan of 1 unit
    flp = flp[np.newaxis, :] 
power_max = args.Pm 
num_steps = args.num_steps
pg_change = args.pg_change

def generate_ptrace_file(flp, num_steps, power_max, filename="random_ptrace.txt"):
    #Generate a power trace file with random power values for each functional unit over time.
    #Output file is in the format: each row = one time step, each column = one functional unit.
    
    Nu = len(flp) #number of functional units
    powers = []
    with open(filename, 'w') as f: #opens file to write
        for t in range(num_steps): #loops through number of time steps
            if t % pg_change == 0:
                powers.clear()
                for _ in range(Nu):
                    powers.append(random.random() * power_max)  #makes a list of powers for each time step
            line = '   '.join(f"{p:.16f}" for p in powers) #formats the powers
            f.write(line + '\n') #writes
    print(f"Power trace file '{filename}' generated with {num_steps} steps and {Nu} units.") #debug print



generate_ptrace_file(flp, num_steps, power_max, filename="random_ptrace.txt")