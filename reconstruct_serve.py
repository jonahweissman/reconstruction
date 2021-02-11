# -*- coding: utf-8 -*-
# -*- mode: python -*-
from __future__ import print_function, division, absolute_import

import os
import glob
import json
import numpy as np
import argparse
import itertools
import ewave
import gammatone.gtgram as gg
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import RidgeCV


def load_pprox(cell, stimuli):
    """ Load stimulus file and response data from local sources """
    unit = os.path.splitext(os.path.basename(cell))[0]
    print(" - responses loaded from:", unit)
    # first load and collate the responses, then load the stimuli
    out = []
    with open(cell, 'r') as fp:
        data = json.load(fp)
        sorter = lambda x: (x['stimulus'], x['condition'], x['trial'])
        grouper = lambda x: (x['stimulus'], x['condition'])
        trials = sorted(data['pprox'], key=sorter)
        for (stimulus, condition), trial in itertools.groupby(trials, grouper):
            stim_name = '_'.join((stimulus,condition))
            try:
                stim_dur = stimuli[stim_name][1]
            except KeyError:
                continue
            out.append({"cell_name": unit,
                        "stimulus": stimulus,
                        "condition": condition,
                        "stim_name": stim_name,
                        "duration": stim_dur,
                        "stim_dt": stimuli[stim_name][2],
                        "spikes": [np.asarray(p["event"]) for p in trial]})
    return out


def load_stimulus(path, window, step, f_min=0.5, f_max=8.0, f_count=30,
                  compress=1, **kwargs):
    """Load sound stimulus and calculate spectrotemporal representation.

    Parameters:

    path: location of wave file
    window: duration of window (in ms)
    step: window step (in ms)
    f_min: minimum frequency (in kHz)
    f_max: maximum frequency (in kHz)
    f_count: number of frequency bins
    gammatone: if True, use gammatone filterbank

    Returns spectrogram, duration (ms)
    """
    fp = ewave.open(path, "r")
    Fs = fp.sampling_rate / 1000.
    osc = ewave.rescale(fp.read(), 'h')
    Pxx = gg.gtgram(osc, Fs * 1000, window / 1000, step / 1000, f_count, f_min * 1000)
    Pxx = np.log10(Pxx + compress) - np.log10(compress)
    return Pxx, Pxx.shape[1] * step, step


def merge_data(seq):
    """Merge a sequence of stimuli into a single trial

    Takes a list or tuple of dicts containing {stim, stim_dt, spike_v, spike_h,
    spike_dt} and concatenates each of {stim, spike_spike_v, spike_h} along the
    appropriate axis, returning a single dictionary with {stim, stim_dt,
    spike_v, spike_h, spike_dt}

    """
    stim_dts = [d["stim_dt"] for d in seq]
    stim_dt = stim_dts[0]
    if not np.all(np.equal(stim_dts, stim_dt)):
        raise ValueError("not all stimuli have the same sampling rate")

    spike_dts = [d["spike_dt"] for d in seq]
    spike_dt = spike_dts[0]
    if not np.all(np.equal(spike_dts, spike_dt)):
        raise ValueError("not all spike vectors have the same sampling rate")

    ntrialss = [d["spike_v"].shape[1] for d in seq]
    ntrials = ntrialss[0]
    if not np.all(np.equal(ntrialss, ntrials)):
        raise ValueError("not all stimuli have the same number of trials")

    spike_v = np.concatenate([d["spike_v"] for d in seq], axis=0)
    data = {
        "stim_dt": stim_dt,
        "spike_dt": spike_dt,
        "ntrials": ntrials,
        "stim_names": [d["stim_name"] for d in seq],
        "psth": np.concatenate([d['psth'] for d in seq]),
        "R": np.concatenate([d["R"] for d in seq], axis=0),
        "spike_v": spike_v,
        "duration": sum(d["duration"] for d in seq),
    }
    return data


def merge_cells(seq,stimuli):
    stim_dts = [d["stim_dt"] for d in seq]
    stim_dt = stim_dts[0]
    if not np.all(np.equal(stim_dts, stim_dt)):
        raise ValueError("not all stimuli have the same sampling rate")

    spike_dts = [d["spike_dt"] for d in seq]
    spike_dt = spike_dts[0]
    if not np.all(np.equal(spike_dts, spike_dt)):
        raise ValueError("not all spike vectors have the same sampling rate")

    ntrialss = [d["spike_v"].shape[0] for d in seq]
    ntrials = ntrialss[0]
    if not np.all(np.equal(ntrialss, ntrials)):
        raise ValueError("not all cells have the same length")

    stimlist = seq[0]["stim_names"]
    if not all(d["stim_names"] == stimlist for d in seq):
        raise ValueError("not all stims are the same")
    stim = np.concatenate([stimuli[name][0] for name in stimlist], axis=1)

    spike_v = np.concatenate([d["spike_v"] for d in seq], axis=1)
    data = {
        "stim_dt": stim_dt,
        "spike_dt": spike_dt,
        "ntrials": ntrials,
        "stim": stim,
        "R": np.concatenate([d["R"] for d in seq], axis=1),
        "spike_v": spike_v,
        "duration": sum(d["duration"] for d in seq),
    }
    return data


def preprocess_spikes(data, dt, nlag=0):
    """Preprocess spike times in data

    Spike times are binned into intervals of duration dt.

    - data: a sequence of dictionaries, which must contain 'spikes', 'stim' and
      'stim_dt' fields.
    - dt: the duration of the step size for the model (same units as spike times)

    The following fields are added in place to the dictionaries in data:

    - spike_v: a 2-D binary array (bins, trials) giving the number of
      spikes in each bin
    - spike_dt: the sampling interval

    """
    for d in data:
        ntrials = len(d["spikes"])
        nframes = d["duration"]
        nbins = nframes * int(d["stim_dt"] / dt) + nlag
        spike_v = np.zeros((nbins, ntrials), dtype='i')
        for i, trial in enumerate(d["spikes"]):
            idx = (trial / dt).astype('i')
            # make sure all spikes are in bounds
            idx = idx[(idx >= 0) & (idx < nbins)]
            spike_v[idx, i] = 1
        d["spike_v"] = spike_v
        d["spike_dt"] = dt
        d["psth"] = np.sum(spike_v, axis=1)
    return data


def split_data(data, fit_group=None, test_group=None):
    if fit_group is not None:
        fit = [d for d in data if {d['condition']}.issubset(fit_group)]
    else:
        fit = data
    if test_group is not None:
        test = [d for d in data if {d['condition']}.issubset(test_group)]
    else:
        test = data
    return fit, test


def r_matrix(data, lag, nb, lin):
    tbmatrix = cosbasis(lag, nb, lin=lin)
    for d in data:
        d['R'] = lagged_matrix(d['psth'], tbmatrix)
    return data


def lagged_matrix(spec, basis):
    """Convert a (nfreq, nt) spectrogram into a design matrix

    basis: can be a positive integer specifying the number of time lags. Or it
    can be a (ntau, nbasis) matrix specifying a set of temporal basis functions
    spanning ntau time lags (for example, the output of cosbasis).

    The output is an (nt, nfreq * nbasis) array. (nbasis = basis when basis is
    an integer)

    """
    from scipy.linalg import hankel

    if spec.ndim == 1:
        spec = np.expand_dims(spec, 0)
    nf, nt = spec.shape
    if np.isscalar(basis):
        ntau = nbasis = basis
    else:
        ntau, nbasis = basis.shape
    X = np.zeros((nt - ntau, nf * nbasis), dtype=spec.dtype)
    for i in range(nf):
        h = np.fliplr(hankel(spec[i, :-ntau], spec[i, -ntau:]))
        if not np.isscalar(basis):
            h = np.dot(h, basis)
        X[:, (i * nbasis):((i + 1) * nbasis)] = h
    return X


def cosbasis(nt, nb, peaks=None, lin=10):
    """Make a nonlinearly stretched basis consisting of raised cosines

    nt:    number of time points
    nb:    number of basis vectors
    peaks: 2-element sequence containing locations of first and last peaks
    lin:   offset for nonlinear stretching of x axis (larger values -> more linear spacing)
    """
    from numpy import cos, clip, pi
    if peaks is None:
        peaks = np.asarray([0, nt * (1 - 1.5 / nb)])

    def nlin(x):
        # nonlinearity for stretching x axis
        return np.log(x + 1e-20)

    y = nlin(peaks + lin)                     # nonlinear transformed first and last
    db = (y[1] - y[0]) / (nb - 1)             # spacing between peaks
    ctrs = np.arange(y[0], y[1] + db, db)     # centers of peaks
    mxt = np.exp(y[1] + 2 * db) - 1e-20 - lin       # maximum time bin
    kt0 = np.arange(0, mxt)
    nt0 = len(kt0)

    def cbas(c):
        return (cos(clip((nlin(kt0 + lin) - c) * pi / db / 2, -pi, pi)) + 1) / 2

    basis = np.column_stack([cbas(c)[::-1] for c in ctrs[::-1]])
    # pad/crop
    if nt0 > nt:
        basis = basis[-nt:]
    elif nt0 < nt:
        basis = np.r_[np.zeros((nt - nt0, nb)), basis]
    # normalize to unit vectors
    basis /= np.linalg.norm(basis, axis=0)
    return basis


def unmerge(prediction, unmerged_data):
    start = 0
    for d in unmerged_data:
        stop = start + d['duration']
        d['predicted'] = prediction[start:stop].T
        start = stop
    return unmerged_data


def plot_reconstructions(data, stimuli, out_dir):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(data), 2, figsize=(10, 20))
    for i, d in enumerate(data):
        stim, duration, dt = stimuli[d['stim_name']]
        axs[i, 0].set_title(' '.join([d['stim_name'], 'actual']))
        axs[i, 0].imshow(stim, aspect="auto", origin="lower")
        axs[i, 1].set_title(' '.join([d['stim_name'], 'predicted']))
        axs[i, 1].imshow(d['predicted'], aspect="auto", origin="lower")
    fig.savefig(os.path.join(out_dir, 'reconstruction.pdf'))


def calc_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


if __name__ == '__main__':
    # Read in data
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--work-dir', help="Directory for intermediate results", default=".")
    parser.add_argument('-s', '--stim-dir', help="Directory with stimulus files", required=True)
    parser.add_argument('-r', '--resp-dir', help="Directory with response files", required=True)
    parser.add_argument('-f', '--file', help='Path to list of units', required=True)
    parser.add_argument('-x', '--xvalidate', help='Cross-validate full parameter set?', default=True)
    parser.add_argument('-b', '--basis', help='Number of basis vectors (if -x False)', default=30)
    parser.add_argument('-l', '--lin', help='Linearity factor (if -x False)', default=30)
    parser.add_argument('-a', '--alpha', help='Alpha value (if -x False)', default=1)
    args = parser.parse_args()

    try:
        stimuli = load(os.path.join(args.work_dir, "stimuli.joblib"))
    except FileNotFoundError:
        print("- loading stimuli from %s" % args.stim_dir)
        stimuli = {}
        for path in glob.glob(os.path.join(args.stim_dir, "*.wav")):
            name = os.path.splitext(os.path.basename(path))[0]
            stimuli[name] = load_stimulus(path, window=2.5, step=1, f_min=1.0, f_max=8.0, f_count=50)
        dump(stimuli, os.path.join(args.work_dir, 'stimuli.joblib'))

    try:
        responses = load(os.path.join(args.work_dir, 'responses.joblib'))
    except FileNotFoundError:
        responses = []
        with open(args.file, 'r') as fp:
            for cell in fp:
                path = os.path.join(args.resp_dir, cell.strip() + ".pprox")
                data = load_pprox(path, stimuli)
                data = preprocess_spikes(data, dt=1, nlag=lag)
                responses.append(data)
        dump(responses, os.path.join(args.work_dir, 'responses.joblib'))

    fit_conditions = ['continuous', 'gap1', 'gap2', 'noise1', 'noise2']
    fit_data = [[r for r in unit if r['condition'] in fit_conditions] for unit in responses]

    # Fit model with cross-validation
    if args.xvalidate is True:
        # Specify parameter values
        params = [(10, 20, 30, 40, 50), (10, 20, 30)]
        lag = 300

        lags = []
        nbs = []
        lins = []
        alphas = []
        scores = []
        aics = []
        best_aic = np.nan
        for nb, lin in itertools.product(*params):
            print("- evaluating model for nb=%d, lin=%d" % (nb, lin))
            fit_data = [r_matrix(f, lag=lag, nb=nb, lin=lin) for f in fit_data]
            fit_merge = merge_cells([merge_data(f) for f in fit_data], stimuli)
            X = fit_merge['R']
            y = fit_merge['stim'].T
            print("  - n=%d, k=%d" % X.shape)
            estimator = RidgeCV(alphas=np.linspace(0.1, 10, 50))
            estimator.fit(X, y)
            lags.append(lag)
            nbs.append(nb)
            lins.append(lin)
            alphas.append(estimator.alpha_)
            scores.append(estimator.best_score_)
            model_aic = calc_aic(len(y), -estimator.best_score_, estimator.coef_.shape[1])
            aics.append(model_aic)
            if np.isnan(best_aic) or best_aic > model_aic:
                best_aic = model_aic
                dump(fit_data, os.path.join(args.work_dir, 'training.joblib'))
                dump(estimator, os.path.join(args.work_dir, 'best_estimator.joblib'))

        # Predict using best estimator
        best_params = pd.DataFrame(list(zip(lags, nbs, lins, alphas, scores, aics)),
                                   columns=["Lag", "NBasis", "Lin", "Alpha", "Score", "AIC"])
        best_params.to_csv(os.path.join(args.work_dir, 'best_params.csv'), index=False)
        best_estimator = load(os.path.join(args.work_dir, 'best_estimator.joblib'))

        lag = best_params.iloc[best_params['AIC'].idxmin()]['Lag']
        nb = best_params.iloc[best_params['AIC'].idxmin()]['NBasis']
        lin = best_params.iloc[best_params['AIC'].idxmin()]['Lin']
    else:
        lag = 300
        nb = args.basis
        lin = args.lin
        print("- fitting model for nb=%d, lin=%d" % (nb, lin))
        fit_data = [r_matrix(f, lag=lag, nb=nb, lin=lin) for f in fit_data]
        fit_merge = merge_cells([merge_data(f) for f in fit_data], stimuli)
        X = fit_merge['R']
        y = fit_merge['stim'].T
        print("  - n=%d, k=%d" % X.shape)
        estimator = RidgeCV(alphas=[args.alpha])
        estimator.fit(X, y)
        dump(fit_data, os.path.join(args.work_dir, 'training.joblib'))
        dump(estimator, os.path.join(args.work_dir, 'best_estimator.joblib'))
        best_estimator = estimator

    # Predict using best estimator
    print("- generating predictions for nb=%d, lin=%d" % (nb, lin))
    test_conditions = ['gapnoise1', 'gapnoise2']
    test_data = [[r for r in unit if r['condition'] in test_conditions] for unit in responses]

    test_data = [r_matrix(t, lag=int(lag), nb=int(nb), lin=int(lin)) for t in test_data]
    test_merge = merge_cells([merge_data(t) for t in test_data], stimuli)
    p = best_estimator.predict(test_merge['R'])

    # Output results
    test = unmerge(p, test_data[0])
    plot_reconstructions(test, stimuli, args.work_dir)
    dump(test_data, os.path.join(args.work_dir, 'testing.joblib'))
    dump(test, os.path.join(args.work_dir, 'predicted.joblib'))
