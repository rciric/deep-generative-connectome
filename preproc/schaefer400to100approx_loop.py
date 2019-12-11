"""
This script computes an approximation of the 100-node Schaefer time series
from the 400-node Schaefer time series.

This is extremely hacky and awful, but it provides a temporary solution that
will allow us to meet our deadline.
"""
import re
import pandas as pd
import pathlib
from collections import defaultdict

path_key = 'key400to100.tsv'
root_400 = 'ts'
root_100 = 'ts/100/{}'

key = pd.read_csv('key400to100.tsv', sep='\t', index_col=0)
ts_400 = pathlib.Path(root_400).glob('*_ts.1D')

key_100 = defaultdict(list)
for i, row in key.iterrows():
    key_100[int(row['parcel'])] += [i]

for f in ts_400:
    print(str(f))
    path = str(f)
    data = pd.read_csv(path, sep=' ', header=None)
    data.columns += 1

    data_100 = pd.DataFrame(0, index=range(len(data)), columns=range(1, 101))
    for i in range(1, 101):
        total_vol = 0
        for j in key_100[i]:
            vol = key.loc[j]['volume']
            total_vol += vol
            data_100[i] += (vol * data[j])
        data_100[i] /= total_vol

    path_out = re.sub('schaefer400', 'schaefer100approx', path.split('/')[-1])
    path_out = root_100.format(path_out)
    data_100.to_csv(path_out, sep='\t')
