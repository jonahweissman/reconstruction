# -*- coding: utf-8 -*-
# -*- mode: python -*-

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


norm = TwoSlopeNorm(vmin=l3diff.min(), vmax = l3diff.min()*-1, vcenter=0)
fig,axes = plt.subplots(5, figsize=(10,10), sharex=True, sharey=True)
axes[0].imshow(cmdiff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
axes[1].imshow(l1diff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
axes[2].imshow(l2adiff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
axes[3].imshow(l3diff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
cb = axes[4].imshow(ncmdiff, origin="lower", aspect="auto", norm=norm, cmap=plt.cm.RdBu_r)
fig.colorbar(cb)
plt.savefig('reconstruction_diffs.pdf')