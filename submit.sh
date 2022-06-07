echo "#!/bin/bash"
echo "#SBATCH --account=def-ka3scott"
echo "#SBATCH --gres=gpu:v100l:2"
echo "#SBATCH --mem=128000"
echo "#SBATCH --time=1-00:00            # time (DD-HH:MM)"
echo "#SBATCH --output=$1"
echo "module load cuda cudnn "
echo "source /home/zgoussea/projects/def-ka3scott/zgoussea/venv/bin/activate"
echo "python /home/zgoussea/projects/def-ka3scott/zgoussea/sifnet/train.py $2"