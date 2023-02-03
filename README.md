# sifnet-implementation

Personal implementation of the SIFNET sea ice forecasting model. This version uses xarray and zarr instead of manipulating numpy arrays and netCDFs directly. Note that the model architecture is unchanged. 

This repository also includes experiments in physically constraining the model by decomposing the change in sea ice concentration into its advection, divergence and residual terms, inspired by https://journals.ametsoc.org/view/journals/clim/29/14/jcli-d-16-0121.1.xml.

TODO: Provide more detail, diagrams, etc. 

### Training a model locally
1. Create a python virtual environment and install the requirements via `pip install -r requirements.txt`
2. Configure the parameters and paths in `main.py`
3. Run the training script: `python main.py --month 0 --predict-fluxes 0 --suffix test`
>Arguments:\
`month`: Month of the year of interest (Default: 0, which trains a model for each month sequentially)\
`predict-fluxes`: [0 (False) or 1 (True)] Whether to train using the physical constraints \
`suffix`: Suffix to append to the output files for run identification \

### Training a model on Compute Canada resources
1. Create a python virtual environment and install the requirements via `pip install -r requirements.txt`
2. Configure the parameters and paths in `main.py`
3. Create a bash file (e.g. `submit_job.sh`) containing the following:

```
#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --gres=gpu:v100l               # Request a single V100 GPU
#SBATCH --mem=128000                   # Request 128GB of memory
#SBATCH --time=00-03:00                # Request a 3-hour allocation (DD-HH:MM)
#SBATCH --output=/path/to/output.log   # File in which logs will be written

module load cuda cudnn                 # Load CUDA modules
source /path/to/env/bin/activate"
python /path/to/sifnet/main.py --month 0 --predict-fluxes 1 --suffix test
```
4. Submit the job via `sbatch submit_job.sh`

### Training all models (for each month) using both the constrained and unconstrained versions
1. Create a python virtual environment and install the requirements via `pip install -r requirements.txt`
2. Configure the parameters and paths in `main.py`
3. Configure the paths in the existing job submission files `submit.sh` and `submit_all.sh`
4. Submit all jobs via `sh submit_all.sh`

Note that this will train 24 models through 24 separate resource allocations. 
