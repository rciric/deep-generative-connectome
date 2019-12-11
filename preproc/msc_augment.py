"""
Data augmentation for MSC connectomes.

When computing the connectome, we bootstrap a subset of frames using
temporal masks.

Sessions 3, 4, 6, 8-10 are training set.
Session 5, 7 are dev set.
Sessions 1-2 are inference.
(Missing acquisitions in sessions 3, 4, 6, 8, 10.)

We augment using a simple 2-way splice across each possible 2-length
permutation of sessions. This will yield sixfold augmentation of the
training set. We're exploiting the same 'sampling variability' that
Laumann and colleagues hold responsible for observed dynamics in the
connectome.

max 36 samples / (subject x task) in the training set
max  4 samples / (subject x task) in the dev and inference sets
(subject x task is 50, so we're looking at up to 1800 for training
and up to 200 for the others.)
"""
import pandas as pd
from numpy import nan
import re
from itertools import permutations

pathout = ('conn/{set}/sub-MSC{sub}_task-{task}_ses-func{ses}'
           '_desc-schaefer100approx_connectome.tsv')

pathgl = ('/oak/stanford/groups/russpold/data/openfmri/ds000224/'
          'sub-MSC{sub}/ses-func{ses}/func/sub-MSC{sub}_ses-func{ses}'
          '_task-glasslexical_run-{run}_events.tsv')
path = ('ts/100/sub-MSC{sub}_ses-func{ses}_task-{task}_run-{run}'
        '_schaefer100approx_ts.1D')
glrx = re.compile('.*_endcue')
glmid = 93 # 96 mid - 3 first volumes discarded
dim = 100

training = ('03', '04', '06', '08', '09', '10')
dev = ('05', '07')
inference = ('01', '02')

sets = (('train', training), ('val', dev), ('test', inference))
sub = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
task = ('rest', 'motor', 'coherence', 'semantic', 'memory')
mem = ('memorywords', 'memoryscenes', 'memoryfaces')


# Get all the permutations for each set.
for u, ses in sets:
    perm = ses + tuple(permutations(ses, 2))
    for s in sub:
        for t in task:
            if t == 'motor' or t =='coherence' or t == 'semantic':
                runs = ('01', '02')
            else:
                runs = ('01',)
            data  = {i: {r: None for r in runs} for i in ses}
            for i in ses:
                for r in runs:
                    # There are two tasks in `glasslexical`:
                    # one half is pattern coherence ('glass') and the other
                    # half is semantic ('lexical'). Figure out which is which
                    # when making the connectome masks.
                    if t == 'coherence' or t =='semantic':
                        tn = 'glasslexical'
                        try:
                            frame = pd.read_csv(
                                path.format(sub=s, task=tn, ses=i, run=r),
                                sep='\t', index_col=0)
                        except FileNotFoundError:
                            continue
                        gl = pathgl.format(sub=s, ses=i, run=r)
                        try:
                            gl = pd.read_csv(gl, sep='\t')
                        except FileNotFoundError:
                            continue
                        gls = gl.loc[gl['trial_type'] == 'Glass_endcue']
                        lex = gl.loc[gl['trial_type'] == 'NV_endcue']
                        if gls['onset'].values < lex['onset'].values:
                            if t == 'coherence':
                                data[i][r] = frame[:glmid]
                            else:
                                data[i][r] = frame[glmid:]
                        else:
                            if t == 'coherence':
                                data[i][r] = frame[glmid:]
                            else:
                                data[i][r] = frame[:glmid]
                    # There are three separate acquisitions for the memory
                    # task.
                    elif t == 'memory':
                        try:
                            data[i][r] = pd.concat([pd.read_csv(
                                path.format(sub=s, task=m, ses=i, run=r),
                                sep='\t', index_col=0) for m in mem])
                        except FileNotFoundError:
                            pass
                    else:
                        try:
                            data[i][r] = pd.read_csv(
                                path.format(sub=s, task=t, ses=i, run=r),
                                sep='\t', index_col=0)
                        except FileNotFoundError:
                            pass
            # Merge across runs
            for k, v in data.items():
                try:
                    data[k] = pd.concat(data[k][i] for i in v.keys())
                except ValueError:
                    data[k] = None
            # Compute all connectome combinations
            for p in perm:
                if isinstance(p, tuple):
                    split = len(p)
                    c = []
                    for i, k in enumerate(p):
                        try:
                            begin = (len(data[k]) * i) // split
                            end = (len(data[k]) * (i + 1)) // split
                            c.append(data[k][begin:end])
                        except TypeError:
                            c.append(pd.DataFrame([nan] * dim))
                    conn = pd.concat(c).corr()
                    p = 'x'.join(['{}' for _ in range(split)]).format(*p)
                else:
                    try:
                        conn = data[p].corr()
                    except AttributeError:
                        conn = pd.DataFrame([nan] * dim)
                print(pathout.format(set=u, sub=s, task=t, ses=p))
                conn.to_csv(pathout.format(set=u, sub=s, task=t, ses=p),
                            sep='\t', header=False, index=False)
