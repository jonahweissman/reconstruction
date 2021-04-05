# -*- coding: utf-8 -*-
# -*- mode: python -*-

import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
import pandas as pd
from scipy.signal import resample

rs = load('RS_probs_margin.joblib')
cs = load('CS_probs_margin.joblib')

rsavg = [np.mean(r, axis=0) for r in rs]
csavg = [np.mean(c, axis=0) for c in cs]

diffs = [cs[i]-rs[i] for i in range(16)]

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

diff_interval = np.empty((len(rs), 50, 110))
diff_interval[:] = np.nan
for i in range(len(diffs)):
    inter = slice(gap['start'].iloc[i] - 5, gap['stop'].iloc[i] + 5)
    diff_interval[i][:][inter] = diffs[i][:][inter]

plt.plot(np.mean(register, axis=0))
# plt.savefig('prob_registered.pdf')