# -*- coding: utf-8 -*-
# -*- mode: python -*-

# Joint posterior probability calculation for multiple reconstruction models
# Requires joblib directory of reconstruction model outputs and ar_prep.py output
# Outputs likelihoods to csv file

import numpy as np
from joblib import load, dump
from scipy import stats
import pandas as pd
import argparse
import os

def interval_prob(Xtildes, ytildex, ytildecn, Vbeta, betahat, s2, gap):
    xs = []
    cn = []
    for i, Xt in enumerate(Xtildes):
        print(" - stim ", i)
        stimxs = np.zeros(len(s2))
        stimcn = np.zeros(len(s2))
        start = gap['start'].iloc[i]
        stop = gap['stop'].iloc[i]
        interval = slice(start, stop)
        center = Xt[interval] @ betahat.T
        variance = np.identity(Xt[interval].shape[0]) + Xt[interval] @ Vbeta @ Xt[interval].T
        for j, sf in enumerate(s2):
            dist = stats.multivariate_normal(center[:, j], sf*variance)
            stimxs[j] = dist.logpdf(ytildex[i][j, interval])
            stimcn[j] = dist.logpdf(ytildecn[i][j, interval])
        xs.append(stimxs)
        cn.append(stimcn)
    return xs, cn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='joblib directory')
    args = parser.parse_args()

    ytilders = load(os.path.join(args.dir, 'ytilders.joblib'))
    ytildecs8 = load(os.path.join(args.dir, 'ytildecs.joblib'))
    ytildecs = [ytildecs8[i // 2] for i in range(len(ytilders))]
    ytildeds = load(os.path.join(args.dir, 'ytildeds.joblib'))
    ytildecn = load(os.path.join(args.dir, 'ytildecn.joblib'))
    if os.path.exists('gaptimes.csv'):
        gap = pd.read_csv('gaptimes.csv')
    else:
        raise ValueError("gaptimes.csv not found")

    areas = ['all', 'cm', 'l1', 'l2a', 'l3', 'ncm']
    area_ll = []
    for i, area in enumerate(areas):
        estimator = load(os.path.join(args.dir, ''.join(['best_estimator_', area ,'.joblib'])))
        Vbeta = load(os.path.join(args.dir, ''.join(['Vbeta_', area, '.joblib'])))
        s2 = load(os.path.join(args.dir, ''.join(['s2_', area, '.joblib'])))
        Xtildes = load(os.path.join(args.dir, ''.join(['Xtildes_', area, '.joblib'])))

        betahat = estimator.coef_

        rs, cn = interval_prob(Xtildes, ytilders, ytildecn, Vbeta, betahat, s2, gap)
        rsfreq = np.asarray([cn[i] - rs[i] for i in range(len(ytilders))])
        rsl = np.sum(rsfreq, axis=1)

        ds, cn = interval_prob(Xtildes, ytildeds, ytildecn, Vbeta, betahat, s2, gap)
        dsfreq = np.asarray([cn[i] - ds[i] for i in range(len(ytilders))])
        dsl = np.sum(dsfreq, axis=1)

        cs, cn = interval_prob(Xtildes, ytildecs, ytildecn, Vbeta, betahat, s2, gap)
        csfreq = np.asarray([cn[i] - cs[i] for i in range(len(ytilders))])
        csl = np.sum(csfreq, axis=1)

        ll = pd.DataFrame({
            "Song": list(range(1, 17)) * 3,
            "Area": [area] * len(ytilders) * 3,
            "Comp": ['Replaced'] * len(ytilders) + ['Discontinuous'] * len(ytildeds) + ['Continuous'] * len(ytildecs),
            "Likelihood": rsl.tolist() + dsl.tolist() + csl.tolist()
        }, index=list(range(i * len(ytilders) * 3, (i+1) * len(ytilders) * 3)))

        area_ll.append(ll)

    area_ll = pd.concat(area_ll)
    area_ll.to_csv('area_ll.csv')



