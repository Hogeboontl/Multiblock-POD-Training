# FEM Training and POD Mode Demo

This repository provides a foundational demonstration of FEM (Finite Element Method) simulations and the generation of POD (Proper Orthogonal Decomposition) modes for a CPU chip for the multi block method. For the multiblock method, the chip is divided into modular “blocks,” each of which can be independently trained. These blocks can then be mathematically combined to form a multi-block reduced-order model. The surrounding regions for each block in this setup are artificially supplied rather than true neighboring blocks so that randomized power inputs can be applied, allowing the resulting POD modes to properly capture variability arising from boundary condition effects.  **Note:** Due to limited manpower, the project was not developed beyond this point and only contains Embarrassingly Parallel optimization for computing multiple blocks at once and is missing significant optimization in regards to the overall FEM and POD calculations.

## Installation

A `.yml` file is included for setting up a Conda environment for a simplified demo experience. Creating an environment from this file will install all necessary dependencies:

```
conda env create -f environment.yml
conda activate <environment_name>
```
Please ensure that the config.py file and the utils folder are present. These contain essential functions required by the training scripts to distribute workloads.

## Input Data Requirements
To run the code, you will need a floorplan and a power trace file that match the structure of the provided examples:

* Floorplan: Each row represents a block to be trained, with columns in the following order: width, height, x_coord, y_coord
* Power Trace: Each column corresponds to a block, and each row represents the power at one time step. The number of rows must match the number of simulation time steps.

If your files are named differently, update the np.loadtxt lines in get_training_data.py accordingly.

## Running the Code

To generate training data, use:

```
mpirun -n X python3 get_training_data.py
```

for the POD modes, use:
```
python3 get_pod_modes.py
```

 plot.py will provide a png of the eigvalues, which can be used to signify which POD modes will provide useful data. It can be run using:
 ```
 python3 plot.py
 ```

## Configuration
All configurable parameters are located in config.py under the config_args section. You can also override these values from the command line using:

```
mpirun -n 16 python3 get_training_data.py --<VARIABLE_NAME> <VALUE>
```

All optional arguments are documented in the config_args section of config.py.


