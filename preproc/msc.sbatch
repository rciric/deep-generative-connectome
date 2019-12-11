#!/bin/bash
#
#SBATCH -J fmriprep
#SBATCH --array=1-10
#SBATCH --time=24:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH -p russpold,owners,normal

# Outputs ----------------------------------
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=%u@stanford.edu
#SBATCH --mail-type=ALL
# ------------------------------------------

unset PYTHONPATH
export SINGULARITYENV_FS_LICENSE=$HOME/ds000224/.freesurfer.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME=/oak/stanford/groups/russpold/data/templateflow/

BIDS_DIR=${OAK}/data/ds000224/
OUTPUT_DIR=${OAK}/data/openfmri/derivatives/ds000224/

SINGULARITY_CMD="singularity run $HOME/ds000224/fmriprep_1.5.1.img"

subject=$( sed -n -E \
    "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" \
    ${BIDS_DIR}/participants.tsv )

cmd="${SINGULARITY_CMD} ${BIDS_DIR} ${OUTPUT_DIR} participant \
    --participant-label $subject \
    -w ${L_SCRATCH}/work/ \
    --omp-nthreads 8 --nthreads 12 --mem_mb 30000 \
    --output-spaces MNI152NLin2009cAsym:res-2 anat \
    --use-syn-sdc"
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?
printf "sub-${subject}\t${SLURM_ARRAY_TASK_ID}\t${exitcode}\n" \
    >> ${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode
