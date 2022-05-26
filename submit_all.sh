for i in {1..12}
do
  OUTPUT="/home/zgoussea/scratch/output$i.out"
  echo $OUTPUT
  sh submit.sh $OUTPUT $i | sbatch
done
