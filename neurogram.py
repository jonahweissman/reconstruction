# -*- coding: utf-8 -*-
# -*- mode: python -*-

import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

train = load('training.joblib')
test = load('testing.joblib')
estimator = load('best_estimator.joblib')


weights = np.mean(estimator.coef_,axis=0)
dweights = [np.mean(weights[i:i+6]) for i in range(0,len(weights),6)]
rsort = np.argsort(dweights)
rsort = [rsort for i in range(837)]
rsort = np.asarray(rsort)
rsort = rsort.T
b189c = [gaussian_filter1d(t[0]['psth'].astype('d'), 3, mode="constant", cval=0.0) for t in train]
b189c = np.asarray(b189c)
b189sortr = np.take_along_axis(b189c, rsort, axis=0)
b189gn = [gaussian_filter1d(t[0]['psth'].astype('d'), 3, mode="constant", cval=0.0) for t in test]
b189gn = np.asarray(b189gn)
b189gnsortr = np.take_along_axis(b189gn, rsort, axis=0)
b189g = [gaussian_filter1d(t[1]['psth'].astype('d'), 3, mode="constant", cval=0.0) for t in train]
b189g = np.asarray(b189g)
b189gsortr = np.take_along_axis(b189g, rsort, axis=0)
fig, axs = plt.subplots(4, sharex=True)
axs[0].imshow(train[0][0]['stim'],aspect='auto',origin='lower')
#axs[0].vlines([748,824],0,50)
axs[1].imshow(b189sortr, cmap="gray_r", origin="lower", aspect="auto")
axs[2].imshow(b189gsortr, cmap="gray_r", origin="lower", aspect="auto")
axs[3].imshow(b189gnsortr, cmap="gray_r", origin="lower", aspect="auto")
plt.tight_layout()
plt.savefig('B189_neurogram.pdf')