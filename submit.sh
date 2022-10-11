echo "#!/bin/bash"
echo "#SBATCH --account=def-ka3scott"
echo "#SBATCH --gres=gpu:v100l"
echo "#SBATCH --mem=128000"
echo "#SBATCH --time=00-3:00            # time (DD-HH:MM)"
echo "#SBATCH --output=$1"
echo "module load cuda cudnn"
echo "source /home/zgoussea/projects/def-ka3scott/zgoussea/venv/bin/activate"
echo "python /home/zgoussea/projects/def-ka3scott/zgoussea/sifnet/main.py --month $2 --predict-fluxes $3 --suffix $4"
# echo "python /home/zgoussea/projects/def-ka3scott/zgoussea/sifnet/main.py --predict-fluxes $2 --suffix $3"

# sh submit.sh "/home/zgoussea/scratch/logs/output_fluxes_nwt.out" 4 1 "fluxes" | sbatch
# sh submit.sh "/home/zgoussea/scratch/logs/output_direct_nwt.out" 4 0 "direct" | sbatch