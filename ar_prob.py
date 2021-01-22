# -*- coding: utf-8 -*-
# -*- mode: python -*-

import numpy as np
from joblib import load, dump
from scipy import stats
import argparse


def conditional_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2):
    rs = []
    cs = []
    for i, Xt in enumerate(Xtildes):
        stimrs = np.zeros((len(s2), Xt.shape[0]))
        stimcs = np.zeros((len(s2), Xt.shape[0]))
        for j, sf in enumerate(s2):
            # Specify multivariate normal distribution
            center = Xt @ betahat[j]
            covm = sf * (np.identity(Xt.shape[0]) + Xt @ Vbeta @ Xt.T)
            for t in range(len(covm)):
                stimrs[j, t], stimcs[j, t] = time_step_prob(t, center, covm, ytilders[i][j], ytildecs[i // 2][j])
        rs.append(stimrs)
        cs.append(stimcs)
    return rs, cs


def time_step_prob(t, center, cm, yrs, ycs):
    # Set up covariance matrix subsets
    # Covariance matrix of the dependent variable
    c11 = cm[t, t]
    # Custom array only containing covariances, not variances
    c12 = np.delete(cm[t, :], t)
    c21 = c12
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
    cprob = stats.norm(conditional_mu, conditional_cov).logcdf(yrs[t])

    # Conditional CS data
    acs = np.delete(ycs, t)
    cs_mu = m1 + c12 @ c22i @ (acs - m2)
    csprob = stats.norm(cs_mu, conditional_cov).logcdf(ycs[t])

    return cprob, csprob


if __name__ == '__main__':
    train_merge = load('probjobs/train_merge.joblib')
    estimator = load('probjobs/best_estimator.joblib')
    Vbeta = load('probjobs/Vbeta.joblib')
    s2 = load('probjobs/s2.joblib')
    Xtildes = load('probjobs/Xtildes.joblib')
    ytilders = load('probjobs/ytilders.joblib')
    ytildecs = load('probjobs/ytildecs.joblib')
    betahat = estimator.coef_

    rs, cs = conditional_prob(Xtildes, ytilders, ytildecs, Vbeta, betahat, s2)

    dump(rs, 'RS_probs.joblib')
    dump(cs, 'CS_probs.joblib')

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
