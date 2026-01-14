from config import parser, config_args
import numpy as np
import os
from mpi4py import MPI
from utils.block_processor import *
#note, this could benefit from creating a hybrid worker setup so multiple ranks can work on one block and from GPU acceleration,as fenicsx supports it

# Create folders (only on rank 0)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

args = parser.parse_args()
file_path = os.path.abspath(args.file_path)

if rank == 0:
    os.makedirs(os.path.join(file_path, "xdmf"), exist_ok=True)
    os.makedirs(os.path.join(file_path, "xdmf_sol"), exist_ok=True)
    os.makedirs(os.path.join(file_path, "matrix_necessities"), exist_ok=True)
comm.Barrier()  # make sure folders exist before other ranks proceed

def compute_power_density(pd, flp, num_steps, thickness):
    for i in range(num_steps):
        for j in range(len(flp)):
            pd[i][j] = pd[i][j] / (flp[j][0] * flp[j][1] * thickness)

#tags for manager-worker set-up 
TAG_READY = 11
TAG_START = 12
TAG_DONE = 13
TAG_STOP = 14
TAG_EXIT = 15

if __name__ == "__main__": #only run the code if this is the file directly run
    # Parse args and load variables
    tol = args.tol
    k_0 = args.k_0
    k_1 = args.k_1
    rho_silicon = args.rho_silicon
    rho_oxide = args.rho_oxide
    c_silicon = args.c_silicon
    c_oxide = args.c_oxide
    active_thickness = args.active_thickness
    silicon_thickness = args.silicon_thickness
    Ta_val = args.Ta
    h = args.h
    power_max = args.Pm
    num_steps = args.num_steps
    sampling_interval = args.sampling_interval
    T = sampling_interval * num_steps
    delt = T / num_steps
    h_c = args.h_c
    pg_change = args.pg_change

    # Load floorplan and power trace
    flp = np.loadtxt('floorplan_AMD.txt')
    if flp.ndim == 1:
        flp = flp[np.newaxis, :] 
    pd = np.loadtxt('random_ptrace.txt')
    if pd.ndim == 1:  # only one block in the trace
        pd = pd[:, np.newaxis]  # shape becomes (num_steps, 1)
    if rank == 0: #compute power density, only needs to happen once
        compute_power_density(pd, flp, num_steps, active_thickness)
        np.save(os.path.join(file_path, "matrix_necessities", "pd.npy"), pd)
    pd = comm.bcast(pd if rank == 0 else None, root=0) #give it to all ranks
 
    n_blocks = len(flp)
    #runs the code in serial if only one or two processes is called
    #if two processes are called, only one is a worker, which is the same as serial
    
    if size <=2:
        if rank == 0:
            for block_index in range(n_blocks):
                process_block_x(
                    block_index, flp, pd, args,
                    T, h, k_0, rho_oxide, c_oxide,
                    tol, rho_silicon, c_silicon, silicon_thickness,
                    k_1, power_max, active_thickness,
                    delt, h_c, Ta_val, MPI.COMM_SELF, pg_change,file_path
                )
            print("All blocks processed in serial")
        raise SystemExit(0)
    
    #actual manager worker set up

    # set rank 0 as manager
    if rank == 0:
        next_task = 0
        closed_workers = 0
        status = MPI.Status()

        while closed_workers < (size-1):
            _ = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
            source = status.Get_source()
            tag=status.Get_tag()
            if tag == TAG_READY or tag== TAG_DONE:
                #Assign task to workers if they are ready
                if next_task < n_blocks:
                    comm.send(next_task,dest=source,tag=TAG_START)
                    next_task+=1
                else: # if no work, get ready to exit
                    comm.send(None,dest=source,tag=TAG_STOP)
            elif tag == TAG_EXIT:
                closed_workers +=1
            else:
                comm.send(None,dest=source,tag=TAG_STOP)
        print("All blocks processed in parallel with dynamic scheduling")

    else:
        #worker loop
        while True:
            #tell the manager the worker is ready
            comm.send(None,dest=0,tag=TAG_READY)
            #wait for start or stop command
            status=MPI.Status()
            task=comm.recv(source=0,tag=MPI.ANY_TAG,status=status)
            tag=status.Get_tag()

            if tag == TAG_START:
                block_index = int(task)
                process_block_x(
                    block_index, flp, pd, args,
                    T, h, k_0, rho_oxide, c_oxide,
                    tol, rho_silicon, c_silicon, silicon_thickness,
                    k_1, power_max, active_thickness,
                    delt, h_c, Ta_val, MPI.COMM_SELF,
                    pg_change,file_path
                )
                comm.send(block_index,dest=0,tag=TAG_DONE)
            elif tag==TAG_STOP:
                comm.send(None,dest=0,tag=TAG_EXIT)
                break
    comm.barrier()







    



