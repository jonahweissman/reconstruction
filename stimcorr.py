# -*- coding: utf-8 -*-
# -*- mode: python -*-
from __future__ import print_function, division, absolute_import

import os
import numpy as np
import pandas as pd
from joblib import load
import argparse


def stim_correlate(pred, gap, stims):
    dcorr = np.zeros((len(pred), 5))
    songs = []
    for i in range(len(pred)):
        song = gap.iloc[i]['song']
        val = gap.iloc[i]['version']
        syllcont = gap[(gap['song'] == song) & (gap['version'] != val)]
        start = gap.iloc[i]['start']
        stop = gap.iloc[i]['stop']
        prediction = pred[i]['predicted'][:, start:stop]
        noise = stims['_'.join((song, ''.join(('gapnoise', str(val)))))][0][:, start:stop]
        continuous = stims['_'.join((song, 'continuous'))][0][:, start:stop]
        gapstim = stims['_'.join((song, ''.join(('gap', str(val)))))][0][:, start:stop]
        cn = continuous + noise
        dcorr[i][0] = dot_corr(prediction, continuous)
        dcorr[i][1] = dot_corr(prediction, gapstim)
        dcorr[i][2] = dot_corr(prediction, noise)
        dcorr[i][3] = dot_corr(prediction, cn)

        scstart = syllcont.iloc[0]['start']
        scstop = syllcont.iloc[0]['stop']
        scpred = pred[i]['predicted'][:, scstart:scstop]
        sccont = stims['_'.join((song, 'continuous'))][0][:, scstart:scstop]
        dcorr[i][4] = dot_corr(scpred, sccont)
        songs.append('_'.join((song, str(val))))

    return dcorr, songs


def dot_corr(x, y):
    mx = x - np.mean(x)
    my = y - np.mean(y)
    n = (mx * my).sum()
    d = (mx * mx).sum() * (my * my).sum()
    dc = n / np.sqrt(d)
    return dc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='joblib directory')
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.dir, 'predicted.joblib')):
        pred = load(os.path.join(args.dir, 'predicted.joblib'))
    else:
        raise ValueError("predicted.joblib not in {}".format(args.dir))

    if os.path.exists('gaptimes.csv'):
        gap = pd.read_csv('gaptimes.csv')
    else:
        raise ValueError("gaptimes.csv not found")

    if os.path.exists(os.path.join(args.dir, 'stimuli.joblib')):
        stims = load(os.path.join(args.dir, 'stimuli.joblib'))
    else:
        raise ValueError("stimuli.joblib not in {}".format(args.dir))
    dcorr, songs = stim_correlate(pred, gap, stims)
    correlations = pd.DataFrame(data=dcorr, index=songs,
                                columns=['Continuous', 'Gap', 'Replaced', 'ContNoise', 'Control'])
    correlations.to_csv('correlations.csv')
