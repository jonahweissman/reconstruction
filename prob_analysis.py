# -*- coding: utf-8 -*-
# -*- mode: python -*-

import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
import pandas as pd
from scipy.signal import resample
from matplotlib.colors import TwoSlopeNorm

rs = load('RS_probs_margin.joblib')
cs = load('CS_probs_margin.joblib')
cn = load('CN_probs_margin.joblib')

rsavg = [np.mean(r, axis=0) for r in rs]
csavg = [np.mean(c, axis=0) for c in cs]

rssum = [np.sum(r, axis=0) for r in rs]
cnsum = [np.sum(c, axis=0) for c in cn]

diffs = [cn[i]-rs[i] for i in range(16)]

davg = [np.mean(d, axis=0) for d in diffs]

gap = pd.read_csv('gaptimes.csv')

register = np.zeros((len(rs), 100))
for i in range(len(register)):
    inter = slice(gap['start'].iloc[i], gap['stop'].iloc[i])
    register[i] = resample(davg[i][inter], 100)

for i in range(4):
    for j in range(4):
        plt.plot(register[j + i*4])

unregister = np.zeros((len(rs), 110))
for i in range(len(unregister)):
    inter = slice(gap['start'].iloc[i]-5, gap['start'].iloc[i]+105)
    unregister[i] = davg[i][inter]

diff_interval = np.empty((len(rs), 50, 100))
diff_interval[:] = np.nan
for i in range(len(diffs)):
    start = gap['start'].iloc[i]
    stop = gap['stop'].iloc[i]
    inter = slice(start, stop)
    diff_interval[i][:, 0:stop-start] = diffs[i][:, inter]

diff_mean = np.nanmean(diff_interval, axis=0)
norm = TwoSlopeNorm(vmin=diff_mean.max()*-1, vmax=diff_mean.max(), vcenter=0)
plt.imshow(diff_mean,origin="lower",aspect="auto",cmap=plt.cm.RdBu_r, norm=norm)

fig,axes = plt.subplots(4, 4, figsize=(10,10))
norm = TwoSlopeNorm(vmin=np.nanmax(diff_interval)*-1, vmax=np.nanmax(diff_interval), vcenter=0)
for i in range(4):
    for j in range(4):
        axes.imshow(diff_interval[j + i*4])

fig,axes = plt.subplots(16, figsize=(5, 10), sharey=True)
for i in range(16):
    axes[i].plot(np.sum(diff_interval[i], axis=0))
    axes[i].hlines(y=0, xmin=0, xmax=100)
plt.savefig('CNRS_lines.pdf')




