# total jank, but we need this up and running quickly.
# we'll go back and do this the right way when we're less pressed for time.

module load R/3.6.1
module load afni/17.3.03
module load ants/2.2.0

SUBPATH='/oak/stanford/groups/russpold/data/openfmri/derivatives/*/fmriprep/sub*'
XCPEDIR='/home/users/rastko/xcpEngine/xcpEngine'
EXCL_THRESH=0.5
images=$(find ${SUBPATH} -name '*bold*MNI152*preproc*.nii*')
PROJDIR=/home/users/rastko/GenerativeConnectome/derivatives/

mkdir -p ${PROJDIR}
printf "dataset\tsubject\tsession\trun\ttask\n" > ${PROJDIR}/exclusion.tsv

for im in $images
    do
    rm -f /tmp/model.1D
    rm -f /tmp/ts.1D
    rm -f /tmp/adjmat.1D
    rm -f /tmp/tmask.1D

    imname=$(echo ${im}|rev|cut -d'/' -f1|rev)
    imdir=$(echo ${im}|rev|cut -d'/' -f2-|rev)
    dsid=$(echo ${imdir}|sed s@'/'@'\n'@g|grep -i '^ds')
    sub=$(echo ${imname}|sed s@'_'@'\n'@g|grep -i '^sub-')
    ses=$(echo ${imname}|sed s@'_'@'\n'@g|grep -i '^ses-')
    run=$(echo ${imname}|sed s@'_'@'\n'@g|grep -i '^run-')
    task=$(echo ${imname}|sed s@'_'@'\n'@g|grep -i '^task-')
    confs=$(ls -d1 ${imdir}/*${sub}*${ses}*${task}*${run}*confounds.tsv)
    mask=$(ls -d1 ${imdir}/*${sub}*${ses}*${task}*${run}*MNI152*mask.nii*)
    [[ -n ${ses} ]] && args="${args} --ses ${ses}"
    [[ -n ${run} ]] && args="${args} --run ${run}"
    [[ -n ${task} ]] && args="${args} --task ${task}"
    outbase=$(python3 output_base.py --sub ${sub} ${args} --out ${PROJDIR}/${dsid})
    mkdir -p ${PROJDIR}/${dsid}
    echo "PROCESSING: ${sub} ${ses} ${run} ${task} from ${dsid}"

    model=($(python3 build_confound_model.py --confs ${confs} --out $TMPDIR/))
    paste ${model[0]} ${model[1]} > /tmp/premodel.1D
    tail -n+2 /tmp/premodel.1D|sed s@'\t'@','@g|sed s@'n/a'@'0'@g >> /tmp/model.1D
    tail -n+2 ${model[2]}|sed s@'^0.*'@'a'@g|sed s@'^1.*'@'0'@g|sed s@'a'@'1'@g >> /tmp/tmask.1D
    num_flagged=$(grep '^0' /tmp/tmask.1D|wc -l)
    num_total=$(cat /tmp/tmask.1D|wc -l)
    pct_flagged=$(echo "scale=10; $num_flagged / $num_total"|bc)
    if [[ $(echo "${pct_flagged} > ${EXCL_THRESH}"|bc) == 1 ]]
        then
        echo "MOTION: excluding ${sub} ${ses} ${run} ${task} from ${dsid}"
        printf "${dsid}\t${sub}\t${ses}\t${run}\t${task}\n" >> /tmp/exclusion.tsv
    fi
    3dTproject  -input ${im} \
        -prefix /tmp/desc-denoised_bold.nii.gz \
        -ort /tmp/model.1D \
        -passband 0.01 0.08 \
        -mask ${mask} \
        -overwrite

    # gotta reorient the atlas since fmriprep uses RAS+ instead of LAS+
    # and ANTs isn't smart enough to figure out the difference.
    3dresample -input ${XCPEDIR}/atlas/schaefer400x7/schaefer400x7MNI.nii.gz \
        -prefix /tmp/schaefer400x7MNI.nii.gz \
        -master /tmp/desc-denoised_bold.nii.gz \
        -overwrite
    ${XCPEDIR}/utils/roi2ts.R -i /tmp/desc-denoised_bold.nii.gz \
        -r /tmp/schaefer400x7MNI.nii.gz \
        -l ${XCPEDIR}/atlas/schaefer400x7/schaefer400x7NodeIndex.1D \
        >> /tmp/ts.1D
    ${XCPEDIR}/utils/ts2adjmat.R -t /tmp/ts.1D \
        -m /tmp/tmask.1D \
        >> /tmp/adjmat.1D

    mv /tmp/adjmat.1D ${outbase}_desc-schaefer400x7_connectome.1D
    mv /tmp/ts.1D ${outbase}_desc-schaefer400x7_timeseries.1D
    mv /tmp/tmask.1D ${outbase}_desc-temporal_mask.1D

done
