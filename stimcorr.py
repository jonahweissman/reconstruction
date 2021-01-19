# -*- coding: utf-8 -*-
# -*- mode: python -*-
from __future__ import print_function, division, absolute_import

import os
import numpy as np
import pandas as pd
from joblib import load
import argparse


def stim_correlate(pred, train, gap, csyll):
    freqcorr = np.zeros((len(pred), 4, 50))
    mcorr = np.zeros((len(pred), 4))
    dcorr = np.zeros((len(pred), 4))
    ccorr = np.zeros((len(pred), 4))
    dcontrol = []
    dsyllcont = []
    for i in range(len(pred)):
        song = gap.iloc[i]['song']
        val = gap.iloc[i]['version']
        control = csyll[csyll['song'] == song]
        syllcont = gap[(gap['song'] == song) & (gap['version'] != val)]
        start = gap.iloc[i]['start']
        stop = gap.iloc[i]['stop']
        prediction = pred[i]['predicted'][:, start:stop]
        noise = pred[i]['stim'][:, start:stop]
        tpos = (i // 2) * 5
        continuous = train[0][tpos]['stim'][:, start:stop]
        gapstim = train[0][tpos + val]['stim'][:, start:stop]
        cn = continuous + noise
        gccorr = np.corrcoef(prediction, continuous)
        gpcorr = np.corrcoef(prediction, gapstim)
        gncorr = np.corrcoef(prediction, noise)
        gcncorr = np.corrcoef(prediction, cn)
        gc = gccorr.diagonal(50)
        gp = gpcorr.diagonal(50)
        gn = gncorr.diagonal(50)
        gcn = gcncorr.diagonal(50)
        freqcorr[i][0] = gc
        mcorr[i][0] = np.mean(gc)
        dcorr[i][0] = dot_corr(prediction, continuous)
        freqcorr[i][1] = gp
        mcorr[i][1] = np.mean(gp)
        dcorr[i][1] = dot_corr(prediction, gapstim)
        freqcorr[i][2] = gn
        mcorr[i][2] = np.mean(gn)
        dcorr[i][2] = dot_corr(prediction, noise)
        freqcorr[i][3] = gcn
        mcorr[i][3] = np.mean(gcn)
        dcorr[i][3] = dot_corr(prediction, cn)

        scstart = syllcont.iloc[0]['start']
        scstop = syllcont.iloc[0]['stop']
        scpred = pred[i]['predicted'][:, scstart:scstop]
        sccont = train[0][tpos]['stim'][:, scstart:scstop]
        dsyllcont.append(dot_corr(scpred, sccont))


        songcontrol = []
        for j in range(len(control)):
            cstart = control.iloc[j]['start']
            cstop = control.iloc[j]['end']
            cpred = pred[i]['predicted'][:, cstart:cstop]
            ccont = train[0][tpos]['stim'][:, cstart:cstop]
            songcontrol.append(dot_corr(cpred, ccont))
            dcontrol.append(songcontrol)

    dcontrol = np.asarray(dcontrol)
    dsyllcont = np.asarray(dsyllcont)

    return freqcorr, mcorr, dcorr, ccorr, dcontrol, dsyllcont


def dot_corr(x,y):
    mx = x-np.mean(x)
    my = y - np.mean(y)
    n = (mx*my).sum()
    d = (mx*mx).sum()*(my*my).sum()
    dc = n/np.sqrt(d)
    return dc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='joblib directory')
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.dir,'predicted.joblib')):
        pred = load(os.path.join(args.dir,'predicted.joblib'))
    else:
        raise ValueError("predicted.joblib not in {}".format(args.dir))
    if os.path.exists(os.path.join(args.dir,'training.joblib')):
        train = load(os.path.join(args.dir,'training.joblib'))
    else:
        raise ValueError("training.joblib not in {}".format(args.dir))
    if os.path.exists('gaptimes.csv'):
        gap = pd.read_csv('gaptimes.csv')
    else:
        raise ValueError("gaptimes.csv not found")
    if os.path.exists('control_syllables.csv'):
        csyll = pd.read_csv('control_syllables.csv')
    else:
        raise ValueError("control_syllables.csv not found")
    freqcorr, mcorr, dcorr, ccorr, dcontrol, dsyllcont = stim_correlate(pred, train, gap, csyll)
    np.savez('correlations.npz', freqcorr=freqcorr, mcorr=mcorr)
