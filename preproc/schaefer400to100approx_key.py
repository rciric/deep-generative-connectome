"""
This script generates a key for mapping the nodes of one parcellation to those
of another parcellation.

Shell code to generate inputs from the Schaefer atlas files:

xcpEngine/xcpEngine/utils/telescope.R \
    -i atlas/schaefer400x7/schaefer400x7MNI.nii.gz \
    -o /tmp/schaefer400telescoped.nii.gz

fslcc /tmp/schaefer400telescoped.nii.gz /tmp/schaefer100telescoped.nii.gz \
    >> crosscorr100to400.txt

3dROIstats \
    -mask xcpEngine/xcpEngine/atlas/schaefer400x7/schaefer400x7MNI.nii.gz \
    -nzvoxels -nomeanout -numROI 400 \
    xcpEngine/xcpEngine/atlas/schaefer400x7/schaefer400x7MNI.nii.gz \
    >> 400x7vols.txt
"""

import pandas as pd
import re

path_cc = 'crosscorr100to400.txt'
path_vols = '400x7vols.txt'
path_key = 'key400to100.tsv'

cc = pd.read_table(path_cc,
                   header=None,
                   sep='\s+',
                   index_col=0)

vol = pd.read_table(path_vols)
vol.columns = [int(re.sub('NZcount_', '', c)) if i >= 2 else c
               for i, c in enumerate(vol.columns)]
vol = vol.transpose()[2:].astype('int64')
vol.columns = ['volume']

map = {i: (None, 0.0) for i in range(1, 401)}
for i, row in cc.iterrows():
    pass
    if row[2] > map[i][1]:
        map[i] = (int(row[1]), row[2])

purity = pd.DataFrame(map).transpose()
purity.columns = ['parcel', 'purity']

key = vol.merge(purity, left_index=True, right_index=True)
key.to_csv(path_key, sep='\t')
