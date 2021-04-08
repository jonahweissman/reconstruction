# -*- coding: utf-8 -*-
# -*- mode: python -*-

import numpy as np
from joblib import load, dump
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='joblib directory')
    args = parser.parse_args()

    # Load joblib files from reconstruction output
    train = load(os.path.join(args.dir, 'training.joblib'))
    test = load(os.path.join(args.dir, 'testing.joblib'))
    stimuli = load(os.path.join(args.dir, 'stimuli.joblib'))
    estimator = load(os.path.join(args.dir, 'best_estimator.joblib'))

    # Calculate X and Xtilde
    X = np.concatenate([np.concatenate([d["R"] for d in t], axis=0) for t in train], axis=1)
    Xtilde = np.concatenate([np.concatenate([d["R"] for d in t], axis=0) for t in test], axis=1)

    # Calculate y, ytilders, and yltildecs
    y_conditions = ['continuous', 'gap1', 'gap2', 'noise1', 'noise2']
    ytilders_conditions = ['gapnoise1', 'gapnoise2']
    ytildecs_conditions = ['continuous']
    ytildeds_conditions = ['gap1', 'gap2']
    y = np.concatenate([stimuli['_'.join((t['stimulus'], t['condition']))][0] for t in train[0] if
                        t['condition'] in y_conditions], axis=1)
    ytilders = [stimuli['_'.join((t['stimulus'], t['condition']))][0] for t in test[0] if
                t['condition'] in ytilders_conditions]
    ytildecs = [stimuli['_'.join((t['stimulus'], t['condition']))][0] for t in train[0] if
                t['condition'] in ytildecs_conditions]
    ytildeds = [stimuli['_'.join((t['stimulus'], t['condition']))][0] for t in train[0] if
                t['condition'] in ytildeds_conditions]
    ytildecn = [ytilders[i]/2 + ytildecs[i // 2]/2 for i in range(len(ytilders))]
    dump(ytilders, os.path.join(args.dir, 'ytilders.joblib'))
    dump(ytildecs, os.path.join(args.dir, 'ytildecs.joblib'))
    dump(ytildecn, os.path.join(args.dir, 'ytildecn.joblib'))
    dump(ytildeds, os.path.join(args.dir, 'ytildeds.joblib'))

    # Calculate Vbeta
    Vbeta = np.linalg.pinv(X.T @ X)
    dump(Vbeta, os.path.join(args.dir, 'Vbeta.joblib'))

    # Calculate s2
    betahat = estimator.coef_
    n = X.shape[0]
    k = X.shape[1]
    s2 = [(1 / (n - k)) * ((yf - X @ betahat[i]).T @ (yf - X @ betahat[i])) for i, yf in enumerate(y)]
    dump(s2, os.path.join(args.dir, 's2.joblib'))

    # Calculate Xtildes
    yind = np.cumsum([t['duration'] for t in test[0]])[:-1]
    Xtildes = np.split(Xtilde, yind)
    dump(Xtildes, os.path.join(args.dir, 'Xtildes.joblib'))
