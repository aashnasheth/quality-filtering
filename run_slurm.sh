# validation data  --output logs \
# at least as many CPUs as GPUs

 jid=$(sbatch \
         --parsable \
         --account=ark \
	     --time=4:00:00 \
	     -p gpu-l40 \
         --gpus=l40:1 \
         --mem 200G \
         -c 4 \
         run.sh)
 echo -n "${jid} "
