# -*- coding: utf-8 -*-
# -*- mode: python -*-
from __future__ import print_function, division, absolute_import

import os
import json
import numpy as np
import argparse
import itertools
import nbank
import ewave
import gammatone.gtgram as gg
import libtfr
import pandas as pd
import toelis as tl
from joblib import dump, load
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.linear_model import RidgeCV
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt


def load_pprox(cell, window, step, stimuli=None, alt_base=None, **specargs):
    """ Load stimulus file and response data from local sources """
    unit = os.path.splitext(os.path.basename(cell))[0]
    print(" - responses loaded from:", unit)
    # first load and collate the responses, then load the stimuli
    out = []
    with open(cell, 'r') as fp:
        data = json.load(fp)
        trials = sorted(data['pprox'], key=lambda x: (x['stimulus'], x['condition'], x['stim_uuid'], x['trial']))
        for (stimulus, condition, stimname), trial in itertools.groupby(trials, lambda x: (
        x['stimulus'], x['condition'], x['stim_uuid'])):
            if stimuli is not None and stimname not in stimuli:
                continue
            stimfile = 'stims/' + '_'.join((stimulus,condition)) + '.wav'
            spec, dur = load_stimulus(stimfile, window, step, **specargs)
            out.append({"cell_name": unit,
                        "stimulus": stimulus,
                        "condition": condition,
                        "stim_uuid": stimname,
                        "duration": dur,
                        "stim": spec,
                        "stim_dt": step,
                        "spikes": [np.asarray(p["event"]) for p in trial]})
    return out


def load_stimulus(path, window, step, f_min=0.5, f_max=8.0, f_count=30,
                  compress=1, gammatone=False, **kwargs):
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
    if gammatone:
        Pxx = gg.gtgram(osc, Fs * 1000, window / 1000, step / 1000, f_count, f_min * 1000)
    else:
        # nfft based on desired number of channels btw f_min and f_max
        nfft = int(f_count / (f_max - f_min) * Fs)
        npoints = int(Fs * window)
        if nfft < npoints:
            raise ValueError("window size {} ({} points) too large for desired freq resolution {}. "
                             "Decrease to {} ms or increase f_count.".format(window, f_count,
                                                                             npoints, nfft / Fs))

        nstep = int(Fs * step)
        taper = np.hanning(npoints)
        mfft = libtfr.mfft_precalc(nfft, taper)
        Pxx = mfft.mtspec(osc, nstep)
        freqs, ind = libtfr.fgrid(Fs, nfft, [f_min, f_max])
        Pxx = Pxx[ind, :]
    if compress is not None:
        Pxx = np.log10(Pxx + compress) - np.log10(compress)
    return Pxx, Pxx.shape[1] * step


def pad_stimuli(data, before, after, fill_value=None):
    """Pad stimuli and adjust spike times in data

    Stimuli are usually preceded and followed by silent periods. This function
    pads the spectrograms with either by the specified fill_value, or by the
    average value in the first and last frame (if fill_value is None)

    Spike times are adjusted so that they are reference to the start of the
    padded stimulus, and all spike times outside the padded interval are
    dropped.

    - data: a sequence of dictionaries, which must contain 'spikes', 'stim' and
      'stim_dt' fields. This is modified in place
    - before: interval to pad before stimulus begins (in units of stim_dt)
    - after: interval to pad after stimulus ends

    NB: this needs to be run BEFORE preprocess_spikes as it will not touch
    spike_v or spike_h.

    """
    for d in data:
        dt = d["stim_dt"]
        n_before = int(before / dt)
        n_after = int(after / dt)

        s = d["stim"]
        nf, nt = s.shape
        fv_before = s[:, 0].mean() if fill_value is None else fill_value
        p_before = fv_before * np.ones((nf, n_before), dtype=s.dtype)
        fv_after = s[:, -1].mean() if fill_value is None else fill_value
        p_after = fv_after * np.ones((nf, n_after), dtype=s.dtype)

        d["stim"] = np.c_[p_before, s, p_after]

        newtl = tl.offset(tl.subrange(d["spikes"], -before, d["duration"] + after), -before)
        d["spikes"] = list(newtl)
        d["duration"] += before + after
    return data


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
        "stim": np.concatenate([d["stim"] for d in seq], axis=1),
        "psth": np.concatenate([d['psth'] for d in seq]),
        "R": np.concatenate([d["R"] for d in seq], axis=0),
        "spike_v": spike_v,
        "duration": sum(d["duration"] for d in seq),
    }
    return data


def merge_cells(seq):
    stim_dts = [d["stim_dt"] for d in seq]
    stim_dt = stim_dts[0]
    if not np.all(np.equal(stim_dts, stim_dt)):
        raise ValueError("not all stimuli have the same sampling rate")

    stims = [d["stim"] for d in seq]
    stim = stims[0]
    if not np.all(np.equal(stims, stim)):
        raise ValueError("not all stims are the same")

    spike_dts = [d["spike_dt"] for d in seq]
    spike_dt = spike_dts[0]
    if not np.all(np.equal(spike_dts, spike_dt)):
        raise ValueError("not all spike vectors have the same sampling rate")

    ntrialss = [d["spike_v"].shape[0] for d in seq]
    ntrials = ntrialss[0]
    if not np.all(np.equal(ntrialss, ntrials)):
        raise ValueError("not all cells have the same length")

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


def preprocess_spikes(data, dt):
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
        nchan, nframes = d["stim"].shape
        nbins = nframes * int(d["stim_dt"] / dt)
        spike_v = np.zeros((nbins, ntrials), dtype='i')
        for i, trial in enumerate(d["spikes"]):
            idx = (trial / dt).astype('i')
            # make sure all spikes are in bounds
            idx = idx[(idx >= 0) & (idx < nbins)]
            spike_v[idx, i] = 1
        d["spike_v"] = spike_v
        d["spike_dt"] = dt
    return data


def make_psth(spike_v, downsample=None, smooth=None):
    """Compute psth from multi-trial spike vector (dimension nbins x ntrials)

    downsample: if not None, factor by which to downsample the PSTH
    smooth:     if not None, smooth the downsampled PSTH
    """

    nbins, ntrials = spike_v.shape
    if downsample is not None:
        new_bins = nbins // downsample
        psth = np.sum(spike_v[:(new_bins * downsample), :].reshape(new_bins, ntrials, -1), axis=(1, 2))
    else:
        psth = np.sum(spike_v, axis=1)
    if smooth is not None:
        return gaussian_filter1d(psth.astype('d'), smooth, mode="constant", cval=0.0)
    else:
        return psth


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
    
    if spec.ndim == 1:
        spec = np.expand_dims(spec, 0)
    nf, nt = spec.shape
    if np.isscalar(basis):
        ntau = nbasis = basis
    else:
        ntau, nbasis = basis.shape
    X = np.zeros((nt, nf * nbasis), dtype=spec.dtype)
    padding = np.zeros(ntau-1, dtype=spec.dtype)
    for i in range(nf):
        h = toeplitz(np.concatenate([spec[i, (ntau-1):], padding]), spec[i, ::-1][-ntau:])
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


def prep_data(cell, window=2.5, step=1, f_min=1.0, f_max=8.0, f_count=50, gammatone=True):
    data = load_pprox(cell, window=window, step=step, f_min=f_min, f_max=f_max, f_count=f_count, gammatone=gammatone)
    data = preprocess_spikes(data, dt=1)
    fit, test = split_data(data, fit_group=['continuous', 'gap1', 'gap2', 'noise1', 'noise2'],
                           test_group=['gapnoise1', 'gapnoise2'])
    for d in fit:
        d['psth'] = make_psth(d['spike_v'])
    for d in test:
        d['psth'] = make_psth(d['spike_v'])
    fit = r_matrix(fit, lag=50, nb=5, lin=10)
    test = r_matrix(test, lag=50, nb=5, lin=10)
    return fit, test


def plot_reconstructions(data):
    fig, axs = plt.subplots(len(data), 2, figsize=(10, 20))
    for i, d in enumerate(data):
        axs[i, 0].set_title(' '.join([d['stimulus'], d['condition'], 'actual']))
        axs[i, 0].imshow(d['stim'], aspect="auto", origin="lower")
        axs[i, 1].set_title(' '.join([d['stimulus'], d['condition'], 'predicted']))
        axs[i, 1].imshow(d['predicted'], aspect="auto", origin="lower")
    fig.savefig('reconstruction.pdf')


def calc_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


def batch_load(unitfile):
    fit_all = []
    test_all = []
    with open(unitfile, 'r') as fp:
        cells = fp.readlines()
        for cell in cells:
            fit, test = prep_data(nbank.get(cell.strip(), registry_url='https://gracula.psyc.virginia.edu/neurobank')+'.pprox')
            fit_all.append(fit)
            test_all.append(test)
    return test, fit_all, test_all


if __name__ == '__main__':
    # Read in data
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Path to list of units', required=True)
    args = parser.parse_args()
    if os.path.exists('training.joblib') and os.path.exists('testing.joblib') and os.path.exists('predicted.joblib'):
        fit_all = load('training.joblib')
        test_all = load('testing.joblib')
        test = load('predicted.joblib')
    else:
        test, fit_all, test_all = batch_load(args.file)

    # Specify parameter values
    params = [(10, 20, 30, 40, 50), (10, 20, 30)]
    lag = 300

    # Fit model
    lags = []
    nbs = []
    lins = []
    alphas = []
    scores = []
    aics = []
    best_aic = np.nan
    for nb, lin in itertools.product(*params):
        for i in range(1):
            fit_all = [r_matrix(f, lag=lag, nb=nb, lin=lin) for f in fit_all]
            fit_merge = merge_cells([merge_data(f) for f in fit_all])
            X = fit_merge['R']
            y = fit_merge['stim'].T
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
                dump(estimator, 'best_estimator.joblib')

    # Predict using best estimator
    best_params = pd.DataFrame(list(zip(lags, nbs, lins, alphas, scores, aics)), 
        columns=["Lag", "NBasis", "Lin", "Alpha", "Score", "AIC"])
    best_estimator = load('best_estimator.joblib')
    lag = best_params.iloc[best_params['AIC'].idxmin()]['Lag']
    nb = best_params.iloc[best_params['AIC'].idxmin()]['NBasis']
    lin = best_params.iloc[best_params['AIC'].idxmin()]['Lin']
    test_all = [r_matrix(t, lag=int(lag), nb=int(nb), lin=int(lin)) for t in test_all]
    test_merge = merge_cells([merge_data(t) for t in test_all])
    p = best_estimator.predict(test_merge['R'])

    # Output results
    test = unmerge(p, test)
    plot_reconstructions(test)
    dump(fit_all, 'training.joblib')
    dump(test_all, 'testing.joblib')
    dump(test, 'predicted.joblib')
    best_params.to_csv('best_params.csv', index=False)
