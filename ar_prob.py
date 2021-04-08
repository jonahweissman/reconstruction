# -*- coding: utf-8 -*-
# -*- mode: python -*-

import numpy as np
from joblib import load, dump
from scipy import stats
import pandas as pd
import argparse
import os


def marginal_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2):
    rs = []
    pvar = []
    cs = []
    for i, Xt in enumerate(Xtildes):
        print("Stim "+str(i))
        stimrs = np.zeros((len(s2), Xt.shape[0]))
        stimpvar = np.zeros((len(s2), Xt.shape[0]))
        stimcs = np.zeros((len(s2), Xt.shape[0]))
        for k, t in enumerate(Xt):
            center = t @ betahat.T
            variance = np.asarray(s2) * (t @ Vbeta @ t.T)
            dist = stats.norm(center, variance)
            stimrs[:, k] = dist.logpdf(ytilders[i][:, k])
            stimpvar[:, k] = variance
            stimcs[:, k] = dist.logpdf(ytildecs[i // 2][:, k])
        rs.append(stimrs)
        pvar.append(stimpvar)
        cs.append(stimcs)
    return rs, pvar, cs


def interval_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2, gap):
    rs = []
    cs = []
    for i, Xt in enumerate(Xtildes):
        print("Stim ", i)
        stimrs = np.zeros(len(s2))
        stimcs = np.zeros(len(s2))
        start = gap['start'].iloc[i]
        stop = gap['stop'].iloc[i]
        interval = slice(start, stop)
        center = Xt[interval] @ betahat.T
        variance = np.identity(Xt[interval].shape[0]) + Xt[interval] @ Vbeta @ Xt[interval].T
        for j, sf in enumerate(s2):
            print(" - spectral band ", j)
            dist = stats.multivariate_normal(center[:, j], sf*variance)
            stimrs[j] = dist.logpdf(ytilders[i][j, interval])
            stimcs[j] = dist.logpdf(ytildecs[i // 2][j, interval])
        rs.append(stimrs)
        cs.append(stimcs)
    return rs, cs


def conditional_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2, span):
    rs = []
    cs = []
    #Xtildes = Xtildes[0:2]
    for i, Xt in enumerate(Xtildes):
        print("Stim "+str(i))
        if len(span) == 2:
            windows = Xt.shape[0]//span[1]
            stimrs = np.zeros((len(s2), windows))
            stimcs = np.zeros((len(s2), windows))
            for j, sf in enumerate(s2):
                # Specify multivariate normal distribution
                center = Xt @ betahat[j]
                covm = sf * (np.identity(Xt.shape[0]) + Xt @ Vbeta @ Xt.T)
                for x, t in enumerate(range(0, len(covm)-(span[0]+1), span[1])):
                    win = slice(t, t+span[0])
                    stimrs[j, x], stimcs[j, x] = time_step_prob(win, center, covm, ytilders[i][j], ytildecs[i // 2][j])
                print(" - spectral band "+str(j))
        else:
            stimrs = np.zeros((len(s2), 1))
            stimcs = np.zeros((len(s2), 1))
        rs.append(stimrs)
        cs.append(stimcs)
    return rs, cs


def time_step_prob(t, center, cm, yrs, ycs):
    # Set up covariance matrix subsets
    # Covariance matrix of the dependent variable
    c11 = cm[t, t]
    # Custom array only containing covariances, not variances
    c12 = np.delete(cm[t, :], t, axis=1)
    c21 = np.delete(cm[:, t], t, axis=0)
    # Covariance matrix of independent variables
    c22 = np.delete(np.delete(cm, t, axis=0), t, axis=1)
    try:
        c22i = np.linalg.inv(c22)
    except:
        c22i = np.linalg.pinv(c22)

    # Means
    m1 = center[t]
    m2 = np.delete(center, t)

    # Conditional RS data
    ars = np.delete(yrs, t)

    conditional_mu = m1 + c12 @ c22i @ (ars - m2)
    conditional_cov = c11 - c12 @ c22i @ c21

    # Conditional log probability
    cprob = stats.multivariate_normal(conditional_mu, conditional_cov).logpdf(yrs[t])

    # Conditional CS data
    acs = np.delete(ycs, t)
    cs_mu = m1 + c12 @ c22i @ (acs - m2)
    csprob = stats.multivariate_normal(cs_mu, conditional_cov).logpdf(ycs[t])

    return cprob, csprob


def plot_all(rs, cs, gap):
    import matplotlib.pyplot as plt
    from scipy.signal import resample
    rsavg = [np.mean(r, axis=0) for r in rs]
    csavg = [np.mean(c, axis=0) for c in cs]
    fig, axes = plt.subplots(8, 2, figsize=(20, 10), sharey=True)
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].vlines(gap['start'].iloc[i*2+j], ymin=-5, ymax=5)
            axes[i, j].vlines(gap['stop'].iloc[i*2+j], ymin=-5, ymax=5)
            axes[i, j].hlines(0, xmin=0, xmax=800)
            axes[i, j].plot(csavg[i * 2 + j] - rsavg[i * 2 + j])
    plt.ylim(-5, 5)
    fig.savefig('prob_all.pdf')

    # register = np.zeros((len(rs), 100))
    # for i in range(len(register)):
    #     inter = slice(gap['start'].iloc[i], gap['stop'].iloc[i])
    #     register[i] = resample(csavg[i][inter] - rsavg[i][inter], 100)
    # plt.plot(np.mean(register, axis=0))
    # plt.savefig('prob_registered.pdf')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='joblib directory')
    parser.add_argument('-w', '--win', help='window size for calculation', default=1)
    parser.add_argument('-s', '--step', help='step size for calculation', default=1)
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

    betahat = estimator.coef_
    span = (int(args.win), int(args.step))

    #rs, cs = interval_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2, gap)
    #dump(rs, 'RS_probs_interval.joblib')
    #dump(cs, 'CS_probs_interval.joblib')

    rs, pvar, cs = marginal_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2)
    dump(rs, 'RS_probs_margin.joblib')
    dump(pvar, 'var_margin.joblib')
    dump(cs, 'CS_probs_margin.joblib')

    #plot_all(rs, cs, gap)




    # train = load('training.joblib')
    # test = load('testing.joblib')
    # estimator = load('best_estimator.joblib')
    # X = load('trainR.joblib')
    # Xtilde = load('testR.joblib')
    # Vbeta = np.linalg.pinv(X.T @ X)
    # n = X.shape[0]
    # k = X.shape[1]
    # y = train_merge['stim']
    # s2 = [(1/(n-k)) * ((yf - X@betahat[i]).T @ (yf - X@betahat[i])) for i,yf in enumerate(y)]
    # XtVbXtT = Xtilde @ Vbeta @ Xtilde.T
    # covm = [s2f * (np.identity(Xtilde.shape[0]) + XtVbXtT) for s2f in s2]
    # yind = np.cumsum([t['duration'] for t in test[0]])[:-1]
    # Xtildes = np.split(Xtilde, yind)
