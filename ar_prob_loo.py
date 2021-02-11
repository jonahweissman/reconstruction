# -*- coding: utf-8 -*-
# -*- mode: python -*-

import numpy as np
from joblib import load, dump
from scipy import stats
import pandas as pd
import argparse
import os
import itertools


def leave_one_out(units, Xtildes, ytilders, ytildecs, Vbeta, betahat, s2, gap):
    unit_diffs = np.zeros((len(units), len(ytilders), len(s2)))
    params = betahat.shape[1] // len(units)
    start = 0
    for i, unit in enumerate(units):
        print(unit)
        stop = start + params
        interval = slice(start, stop)
        Xoo = [np.delete(Xt, interval, axis=1) for Xt in Xtildes]
        bhoo = np.delete(betahat, interval, axis=1)
        Vboo = np.delete(np.delete(Vbeta, interval, axis=0), interval, axis=1)
        rs, cs = interval_prob(Xoo, ytilders, ytildecs, Vboo, bhoo, s2, gap)
        unit_diffs[i] = np.asarray([cs[i]-rs[i] for i in range(len(ytilders))])
        start += params
    return unit_diffs


def marginal_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2):
    rs = []
    cs = []
    for i, Xt in enumerate(Xtildes):
        print("Stim "+str(i))
        stimrs = np.zeros((len(s2), Xt.shape[0]))
        stimcs = np.zeros((len(s2), Xt.shape[0]))
        for k, t in enumerate(Xt):
            center = t @ betahat.T
            variance = np.asarray(s2) * (t @ Vbeta @ t.T)
            dist = stats.norm(center, variance)
            stimrs[:, k] = dist.logpdf(ytilders[i][:, k])
            stimcs[:, k] = dist.logpdf(ytildecs[i // 2][:, k])
        rs.append(stimrs)
        cs.append(stimcs)
    return rs, cs


def interval_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2, gap):
    rs = []
    cs = []
    for i, Xt in enumerate(Xtildes):
        print(" - stim ", i)
        stimrs = np.zeros(len(s2))
        stimcs = np.zeros(len(s2))
        start = gap['start'].iloc[i]
        stop = gap['stop'].iloc[i]
        interval = slice(start, stop)
        center = Xt[interval] @ betahat.T
        variance = np.identity(Xt[interval].shape[0]) + Xt[interval] @ Vbeta @ Xt[interval].T
        for j, sf in enumerate(s2):
            dist = stats.multivariate_normal(center[:, j], sf*variance)
            stimrs[j] = dist.logpdf(ytilders[i][j, interval])
            stimcs[j] = dist.logpdf(ytildecs[i // 2][j, interval])
        rs.append(stimrs)
        cs.append(stimcs)
    return rs, cs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='joblib directory')
    args = parser.parse_args()

    estimator = load(os.path.join(args.dir, 'best_estimator.joblib'))
    Vbeta = load(os.path.join(args.dir, 'Vbeta.joblib'))
    s2 = load(os.path.join(args.dir, 's2.joblib'))
    Xtildes = load(os.path.join(args.dir, 'Xtildes.joblib'))
    ytilders = load(os.path.join(args.dir, 'ytilders.joblib'))
    ytildecs = load(os.path.join(args.dir, 'ytildecs.joblib'))
    if os.path.exists('gaptimes.csv'):
        gap = pd.read_csv('gaptimes.csv')
    else:
        raise ValueError("gaptimes.csv not found")
    if os.path.exists('unitfiles_all.txt'):
        with open('unitfiles_all.txt', 'r') as fp:
            units = [cell.strip() for cell in fp]

    betahat = estimator.coef_

    # leave-one-out array
    unit_diffs = leave_one_out(units, Xtildes, ytilders, ytildecs, Vbeta, betahat, s2, gap)
    dump(unit_diffs, os.path.join(args.dir, 'unit_diffs.joblib'))

    # leave-none-out array
    rs, cs = interval_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2, gap)
    dfreq = np.asarray([cs[i]-rs[i] for i in range(len(ytilders))])

    # difference the arrays and write to csv
    udnorm = np.asarray([dfreq-u for u in unit_diffs])
    m, n, r = udnorm.shape
    out_arr = np.column_stack((np.repeat(np.arange(m), n), udnorm.reshape(m * n, -1)))
    unit_freq = pd.DataFrame(out_arr)
    songnames = ['_'.join((gap['song'].iloc[i], str(gap['version'].iloc[i]))) for i in range(16)]
    idx = list(itertools.product(*[units, songnames]))
    idx = pd.DataFrame(idx)
    unit_freq = pd.concat([idx, unit_freq], axis=1)
    unit_freq.to_csv(os.path.join(args.dir, 'unit_freq.csv'))
