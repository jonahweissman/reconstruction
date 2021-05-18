# -*- coding: utf-8 -*-
# -*- mode: python -*-

import os
import numpy as np
from joblib import load, dump
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

full = load('probjobs/testing.joblib')
fullp = full[0][7]['predicted']
cm = load('leaveoneout/testing_cm.joblib')
cmp = cm[0][7]['predicted']
cmdiff = fullp-cmp
l1 = load('leaveoneout/testing_l1.joblib')
l1p = l1[0][7]['predicted']
l1diff = fullp-l1p
l2a = load('leaveoneout/testing_l2a.joblib')
l2ap = l2a[0][7]['predicted']
l2adiff = fullp-l2ap
l3 = load('leaveoneout/testing_l3.joblib')
l3p = l3[0][7]['predicted']
l3diff = fullp-l3p
ncm = load('leaveoneout/testing_ncm.joblib')
ncmp = ncm[0][7]['predicted']
ncmdiff = fullp-ncmp


norm = TwoSlopeNorm(vmin=l3diff.max()*-1, vmax=l3diff.max(), vcenter=0)
fig,axes = plt.subplots(5, figsize=(10,10), sharex=True, sharey=True)
axes[0].imshow(cmdiff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
axes[1].imshow(l1diff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
axes[2].imshow(l2adiff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
axes[3].imshow(l3diff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
cb = axes[4].imshow(ncmdiff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
fig.colorbar(cb)
plt.savefig('reconstruction_diffs.pdf')

train = load('probjobs/training.joblib')
test = load('probjobs/testing.joblib')
gap = pd.read_csv('gaptimes.csv')

cresp = np.zeros((407, 16))
for i in range(407):
    k = 0
    for j in range(40):
        if train[i][j]['condition'] == "continuous":
            start = gap['start'][gap['song']==train[i][j]['stimulus']].iloc[0]
            stop = gap['stop'][gap['song']==train[i][j]['stimulus']].iloc[0]
            cresp[i][k] = np.sum(train[i][j]['psth'][start:stop]) / (stop - start) * 100
            k = k + 1
            start = gap['start'][gap['song'] == train[i][j]['stimulus']].iloc[1]
            stop = gap['stop'][gap['song'] == train[i][j]['stimulus']].iloc[1]
            cresp[i][k] = np.sum(train[i][j]['psth'][start:stop]) / (stop - start) * 100
            k = k + 1
cresp = np.mean(cresp, axis=1) + 1

gresp = np.zeros((407, 16))
for i in range(407):
    k = 0
    for j in range(40):
        if train[i][j]['condition'] == "gap1":
            start = gap['start'][gap['song']==train[i][j]['stimulus']].iloc[0]
            stop = gap['stop'][gap['song']==train[i][j]['stimulus']].iloc[0]
            gresp[i][k] = np.sum(train[i][j]['psth'][start:stop]) / (stop - start) * 100
            k = k+1
        elif train[i][j]['condition'] == "gap2":
            start = gap['start'][gap['song'] == train[i][j]['stimulus']].iloc[1]
            stop = gap['stop'][gap['song'] == train[i][j]['stimulus']].iloc[1]
            gresp[i][k] = np.sum(train[i][j]['psth'][start:stop]) / (stop - start) * 100
            k = k+1
gresp = np.mean(gresp, axis=1) + 1

nresp = np.zeros((407, 16))
for i in range(407):
    k = 0
    for j in range(40):
        if train[i][j]['condition'] == "noise1":
            start = gap['start'][gap['song']==train[i][j]['stimulus']].iloc[0]
            stop = gap['stop'][gap['song']==train[i][j]['stimulus']].iloc[0]
            nresp[i][k] = np.sum(train[i][j]['psth'][start:stop]) / (stop - start) * 100
            k = k+1
        elif train[i][j]['condition'] == "noise2":
            start = gap['start'][gap['song'] == train[i][j]['stimulus']].iloc[1]
            stop = gap['stop'][gap['song'] == train[i][j]['stimulus']].iloc[1]
            nresp[i][k] = np.sum(train[i][j]['psth'][start:stop]) / (stop - start) * 100
            k = k+1
nresp = np.mean(nresp, axis=1) + 1

gnresp = np.zeros((407, 16))
for i in range(407):
    k = 0
    for j in range(16):
        if test[i][j]['condition'] == "gapnoise1":
            start = gap['start'][gap['song']==train[i][j]['stimulus']].iloc[0]
            stop = gap['stop'][gap['song']==train[i][j]['stimulus']].iloc[0]
            gnresp[i][k] = np.sum(train[i][j]['psth'][start:stop]) / (stop - start) * 100
            k = k+1
        elif test[i][j]['condition'] == "gapnoise2":
            start = gap['start'][gap['song'] == train[i][j]['stimulus']].iloc[1]
            stop = gap['stop'][gap['song'] == train[i][j]['stimulus']].iloc[1]
            gnresp[i][k] = np.sum(train[i][j]['psth'][start:stop]) / (stop - start) * 100
            k = k+1
gnresp = np.mean(gnresp, axis=1) + 1

if os.path.exists('unitfiles_all.txt'):
    with open('unitfiles_all.txt', 'r') as fp:
        units = [cell.strip() for cell in fp]

unit_rates = pd.DataFrame([units, cresp, nresp, gresp, gnresp]).transpose()
unit_rates.columns = ['ID','CS','NS','DS','RS']
unit_rates.to_csv('unit_rates.csv')