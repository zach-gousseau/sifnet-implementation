for i in {1..12}
do
  OUTPUT="/home/aadriano/scratch/logs/output_direct_$i.out"
  echo $OUTPUT
  sh submit.sh $OUTPUT $i 0 "direct" | sbatch
done

for i in {1..12}
do
  OUTPUT="/home/aadriano/scratch/logs/output_fluxes_$i.out"
  echo $OUTPUT
  sh submit.sh $OUTPUT $i 1 "fluxes" | sbatch
done

: '
for i in {1..12}; do tail -1 /home/aadriano/scratch/logs/output_fluxes_$i.out; done
for i in {1..12}; do tail -1 /home/aadriano/scratch/logs/output_direct_$i.out; done
'
