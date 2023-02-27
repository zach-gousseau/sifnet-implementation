#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --gres=gpu:v100l               # Request a single V100 GPU 
#SBATCH --mem=187000                   # Request 128GB of memory
#SBATCH --time=00-12:00                # Request a 3-hour allocation (DD-HH:MM)
#SBATCH --output=/home/aadriano/scratch/test_run.log   # File in which logs will be written

module load cuda cudnn python/3.8
source /home/aadriano/sifnet/bin/activate
python /home/aadriano/projects/def-ka3scott/aadriano/sifnet-implementation/main.py --month 1 --predict-fluxes 1 --suffix test
