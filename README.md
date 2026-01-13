Installation Guide:

A .yml file is included, using this to create a conda environment will allow for the code to run.

please ensure that the "config.py", and "utils" folder are present. these files have the necessary functions required for the training file to distribute the workload. 


How to run the code:

To run the code, ensure you have a floorplan and power trace file that match the ones from the example in structure. More specifically, the floorplan has rows for each block that needs training, with the columns being width, height, x coord, y coord from left to right. The powertrace has a column for each block in need of training, and each row value is the power trace for one time step. This means the power_trace row indices will need to match the number of time steps that are to be simulated.  if they are named differently, they will need to be adjusted in the "get_training_data.py" file. specifically, adjust the "np.loadtxt" lines. 

All of the necessary data that will be created to run the code will be generated and stored in the "file_path" absolute directory. The powertrace and floorplan are the exception as they are user generated, and can be stored in the main directory path in which the "get_training_data" and "predict" file is stored.

to run the code that will get the training data, please run "mpirun -n x python3 get_training_data.py" in which x should be the number of available cores for the process. Please note, due to a dynamic scheduling master-worker set-up in the code, running with either the values "1" or "2" will result in serial code running. This also means that to run 11 blocks, 12 ranks will need to be called.

to make adjustments to the code, please view the config file, in which you can adjust the values under the "config_args" section, optionally you can also set the values in the command line before running by using "-- PLACEHOLDER_VARIABLE_NAME PLACEHOLDER VALUE" after typing in the portion to run it. this would like like:

mpirun -n 16 --x_dim 10

all of the optional arguments are present in the config file under the "config_args" section





