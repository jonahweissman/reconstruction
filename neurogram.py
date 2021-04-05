# -*- coding: utf-8 -*-
# -*- mode: python -*-

import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
from scipy.stats import zscore

train = load('probjobs/training.joblib')
test = load('probjobs/testing.joblib')
estimator = load('probjobs/best_estimator.joblib')
unitloc = pd.read_csv('restlist.csv')
stimuli = load('probjobs/stimuli.joblib')
gaps = pd.read_csv('gaptimes.csv')

weights = np.mean(estimator.coef_, axis=0)
dweights = [np.mean(weights[i:i + 30]) for i in range(0, len(weights), 30)]
unitloc['Index'] = np.arange(407)
unitloc['Dweights'] = dweights
unitloc['LocSort'] = 0
unitloc.loc[unitloc['Location.y'] == "L1", 'LocSort'] = 1
unitloc.loc[unitloc['Location.y'] == "L2a", 'LocSort'] = 2
unitloc.loc[unitloc['Location.y'] == "L3", 'LocSort'] = 3
unitloc.loc[unitloc['Location.y'] == "NCM", 'LocSort'] = 4
unitloc.loc[pd.isna(unitloc['Location.y']), 'LocSort'] = 5
unitsort = unitloc.sort_values(["LocSort", "Dweights"], ascending=(True, False))

r253_1c = stimuli['R253_continuous']
o129_2c = stimuli['O129_continuous']
b189_1c = stimuli['B189_continuous']

div = [56, 25, 33, 59, 90]
div = np.cumsum(np.asarray(div))

# R253_1
rsort = unitsort["Index"]
rsort = [rsort for i in range(r253_1c[1])]
rsort = np.asarray(rsort)
rsort = rsort.T
cont = [gaussian_filter1d(t[30]['psth'][:r253_1c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in train]
cont = np.asarray(cont)
contsort = np.take_along_axis(cont, rsort, axis=0)
contsort = zscore(contsort, axis=1)
where_are_NaNs = np.isnan(contsort)
contsort[where_are_NaNs] = 0
replaced = [gaussian_filter1d(t[12]['psth'][:r253_1c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in test]
replaced = np.asarray(replaced)
repsort = np.take_along_axis(replaced, rsort, axis=0)
repsort = zscore(repsort, axis=1)
where_are_NaNs = np.isnan(repsort)
repsort[where_are_NaNs] = 0
gap = [gaussian_filter1d(t[31]['psth'][:r253_1c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in train]
gap = np.asarray(gap)
gapsort = np.take_along_axis(gap, rsort, axis=0)
gapsort = zscore(gapsort, axis=1)
where_are_NaNs = np.isnan(gapsort)
gapsort[where_are_NaNs] = 0

fig, axs = plt.subplots(4, figsize=(10, 10),  sharex=True)
axs[0].imshow(r253_1c[0], aspect='auto', origin='lower')
axs[0].vlines([512, 612], 0, 50)
axs[1].imshow(contsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[1].hlines(div, 0, r253_1c[1])
axs[1].vlines([512, 612], 0, 407)
axs[2].imshow(gapsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[2].hlines(div, 0, r253_1c[1])
axs[2].vlines([512, 612], 0, 407)
axs[3].imshow(repsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[3].hlines(div, 0, r253_1c[1])
axs[3].vlines([512, 612], 0, 407)
plt.tight_layout()
plt.savefig('R253_1_neurogram.pdf')

# O129_2
rsort = unitsort["Index"]
rsort = [rsort for i in range(o129_2c[1])]
rsort = np.asarray(rsort)
rsort = rsort.T
cont = [gaussian_filter1d(t[20]['psth'][:o129_2c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in train]
cont = np.asarray(cont)
contsort = np.take_along_axis(cont, rsort, axis=0)
contsort = zscore(contsort, axis=1)
where_are_NaNs = np.isnan(contsort)
contsort[where_are_NaNs] = 0
replaced = [gaussian_filter1d(t[9]['psth'][:o129_2c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in test]
replaced = np.asarray(replaced)
repsort = np.take_along_axis(replaced, rsort, axis=0)
repsort = zscore(repsort, axis=1)
where_are_NaNs = np.isnan(repsort)
repsort[where_are_NaNs] = 0
gap = [gaussian_filter1d(t[22]['psth'][:o129_2c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in train]
gap = np.asarray(gap)
gapsort = np.take_along_axis(gap, rsort, axis=0)
gapsort = zscore(gapsort, axis=1)
where_are_NaNs = np.isnan(gapsort)
gapsort[where_are_NaNs] = 0

fig, axs = plt.subplots(4, figsize=(10, 10),  sharex=True)
axs[0].imshow(o129_2c[0], aspect='auto', origin='lower')
axs[0].vlines([582, 682], 0, 50)
axs[1].imshow(contsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[1].hlines(div, 0, o129_2c[1])
axs[1].vlines([582, 682], 0, 407)
axs[2].imshow(gapsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[2].hlines(div, 0, o129_2c[1])
axs[2].vlines([582, 682], 0, 407)
axs[3].imshow(repsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[3].hlines(div, 0, o129_2c[1])
axs[3].vlines([582, 682], 0, 407)
plt.tight_layout()
plt.savefig('O129_2_neurogram.pdf')

# B189_1
rsort = unitsort["Index"]
rsort = [rsort for i in range(b189_1c[1])]
rsort = np.asarray(rsort)
rsort = rsort.T
cont = [gaussian_filter1d(t[0]['psth'][:b189_1c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in train]
cont = np.asarray(cont)
contsort = np.take_along_axis(cont, rsort, axis=0)
contsort = zscore(contsort, axis=1)
where_are_NaNs = np.isnan(contsort)
contsort[where_are_NaNs] = 0
replaced = [gaussian_filter1d(t[0]['psth'][:b189_1c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in test]
replaced = np.asarray(replaced)
repsort = np.take_along_axis(replaced, rsort, axis=0)
repsort = zscore(repsort, axis=1)
where_are_NaNs = np.isnan(repsort)
repsort[where_are_NaNs] = 0
gap = [gaussian_filter1d(t[1]['psth'][:b189_1c[1]].astype('d'), 3, mode="constant", cval=0.0) for t in train]
gap = np.asarray(gap)
gapsort = np.take_along_axis(gap, rsort, axis=0)
gapsort = zscore(gapsort, axis=1)
where_are_NaNs = np.isnan(gapsort)
gapsort[where_are_NaNs] = 0

fig, axs = plt.subplots(4, figsize=(10, 10),  sharex=True)
axs[0].imshow(b189_1c[0], aspect='auto', origin='lower')
axs[0].vlines([113, 209], 0, 50)
axs[1].imshow(contsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[1].hlines(div, 0, b189_1c[1])
axs[1].vlines([113, 209], 0, 407)
axs[2].imshow(gapsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[2].hlines(div, 0, b189_1c[1])
axs[2].vlines([113, 209], 0, 407)
axs[3].imshow(repsort, cmap="gray_r", origin="upper", aspect="auto", vmin=0, vmax=6)
axs[3].hlines(div, 0, b189_1c[1])
axs[3].vlines([113, 209], 0, 407)
plt.tight_layout()
plt.savefig('B189_1_neurogram.pdf')