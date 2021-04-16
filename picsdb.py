#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Analysis of the:
# Preterm Infant Cardio-Respiratory Signals Database (picsdb)
# Download: PICS database (DOI: https://doi.org/10.13026/C2QQ2M)
# FvW 06/2020

import os
import numpy as np
from scipy.signal import welch, butter, filtfilt
#import matplotlib
import matplotlib.pyplot as plt
import wfdb
import xlrd


def bp_filter(x, fs, f_lo, f_hi):
    """
    implement digital band-pass filter
    6-th order Butterworth filter, zero-phase implementation
    """
    f_Ny = fs/2
    b_lo = f_lo / f_Ny
    b_hi = f_hi / f_Ny
    # filter parameters
    p_lp = {'N':6, 'Wn': b_hi, 'btype': 'lowpass', 'analog': False, 'output': 'ba'}
    p_hp = {'N':6, 'Wn': b_lo, 'btype': 'highpass', 'analog': False, 'output': 'ba'}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    x_filt = filtfilt(bp_b1, bp_a1, x, axis=0)
    x_filt = filtfilt(bp_b2, bp_a2, x_filt, axis=0)
    return x_filt


def interval_stats(ts,dt):
    """
    Inputs:
    ts, array of time points, e.g. local maxima of oscillating curve
    dt: sampling interval in seconds
    """
    dts = np.diff(ts)*dt
    mn = dts.min()
    mx = dts.max()
    mu = dts.mean()
    mu_freq = 1/mu
    sd = dts.std()
    q25 = np.percentile(dts, 25)
    q75 = np.percentile(dts, 75)
    print(f"Interval stats (n={len(dts):d}) :")
    print(f"min: {mn:.2f} sec")
    print(f"max: {mx:.2f} sec")
    print(f"mean: {mu:.2f} sec (= {mu_freq:.2f} Hz)")
    print(f"std: {sd:.2f} sec")
    print(f"q25: {q25:.2f} sec")
    print(f"q75: {q75:.2f} sec")
    return dts


def load_segments(filename, verbose=True):
    """
    Load filter and threshold parameters from .xlsx file
    dictionary of subjects and segments, processing parameters
    """
    w = xlrd.open_workbook(filename)
    R = w.sheet_by_name("Sheet1")
    #print(f"Excel document: rows = {R.nrows:d}, cols = {R.ncols:d}")
    # get number of segments
    n_segments = R.nrows-1
    segments = {}
    subj_set= set() # count unique subject IDs
    for i in range(1, n_segments+1):
        a, b = str(R.cell_value(i,0)).split(".")
        i_subj = int(a)
        i_seg = int(b)
        if verbose:
            print(f"\nsubject: {i_subj:d}, segment: {i_seg:d}")
        key_subj = f"infant{i_subj:d}"
        key_seg = f"segment{i_seg:d}"
        if i_subj not in subj_set:
            segments[key_subj] = {} # start new sub-dictionary for each subject
        segments[key_subj][key_seg] = {}
        subj_set.add(i_subj)
        # segment start (on) and stop (off) indices
        segments[key_subj][key_seg]['on'] = int(R.cell_value(i,1))
        segments[key_subj][key_seg]['off'] = int(R.cell_value(i,2))
        # ECG band-pass frequencies
        segments[key_subj][key_seg]['freq_lo_ecg'] = float(R.cell_value(i,3))
        segments[key_subj][key_seg]['freq_hi_ecg'] = float(R.cell_value(i,4))
        # RESP band-pass frequencies
        segments[key_subj][key_seg]['freq_lo_resp'] = float(R.cell_value(i,5))
        segments[key_subj][key_seg]['freq_hi_resp'] = float(R.cell_value(i,6))
        # ECG & RESP thresholds
        segments[key_subj][key_seg]['thr_ecg'] = float(R.cell_value(i,7))
        segments[key_subj][key_seg]['thr_resp'] = float(R.cell_value(i,8))
        # maximum peak-to-peak frequency for ECG & RESP
        segments[key_subj][key_seg]['f_max_ecg'] = float(R.cell_value(i,9))
        segments[key_subj][key_seg]['f_max_resp'] = float(R.cell_value(i,10))
        if verbose:
            print(f"on: {segments[key_subj][key_seg]['on']:d}")
            print(f"off: {segments[key_subj][key_seg]['off']:d}")
            print(f"freq_lo_ecg: {segments[key_subj][key_seg]['freq_lo_ecg']:.3f}")
            print(f"freq_hi_ecg: {segments[key_subj][key_seg]['freq_hi_ecg']:.3f}")
            print(f"freq_lo_resp: {segments[key_subj][key_seg]['freq_lo_resp']:.3f}")
            print(f"freq_hi_resp: {segments[key_subj][key_seg]['freq_hi_resp']:.3f}")
            print(f"thr_ecg: {segments[key_subj][key_seg]['thr_ecg']:.3f}")
            print(f"thr_resp: {segments[key_subj][key_seg]['thr_resp']:.3f}")
            print(f"f_max_ecg: {segments[key_subj][key_seg]['f_max_ecg']:.3f}")
            print(f"f_max_resp: {segments[key_subj][key_seg]['f_max_resp']:.3f}")
    return segments


def load_waveforms(data_dir, infant_index, verbose=True):
    """
    load ECG/RESP waveforms, return as arrays, return sampling freqs
    """
    file_ecg = f"infant{infant_index:d}_ecg"
    file_resp = f"infant{infant_index:d}_resp"
    record_ecg = wfdb.rdrecord(os.path.join(data_dir, file_ecg))
    record_resp = wfdb.rdrecord(os.path.join(data_dir, file_resp))
    d_ecg = record_ecg.__dict__
    d_resp = record_resp.__dict__
    fs_ecg = d_ecg['fs'] # ECG sampling rate in Hz
    fs_resp = d_resp['fs'] # RESP sampling rate in Hz
    x_ecg = d_ecg['p_signal'].ravel()
    x_resp = d_resp['p_signal'].ravel()
    if verbose:
        print("Loading ECG file: ", file_ecg)
        print("Loading RESP file: ", file_resp)
        #print("ECG record: ", d_ecg['record_name'])
        #print("number of signals: ", d_ecg['n_sig'])
        #print("sampling frequency: ", d_ecg['fs'], "Hz")
        #print("Number of samples: ", d_ecg['sig_len'])
        #print("Signal name: ", d_ecg['sig_name'])
        #print("RESP record: ", d_resp['record_name'])
        #print("number of signals: ", d_resp['n_sig'])
        #print("sampling frequency: ", d_resp['fs'], "Hz")
        #print("Number of samples: ", d_resp['sig_len'])
        #print("Signal name: ", d_resp['sig_name'])
        print("ECG sampling frequency: ", fs_ecg, " Hz")
        print("RESP sampling frequency: ", fs_resp, " Hz")
    return x_ecg, x_resp, fs_ecg, fs_resp


def locmax(x):
    p = 1 + np.where(np.diff(np.sign(np.diff(x))) == -2)[0]
    return p


def p_hat(x, n_bins=50, method='histogram'):
    """
    Histogram / kernel estimate of p(r)
    method = 'histogram', 'kernel'
    """
    x_max = x.max()
    if method == 'histogram':
        x_bins = np.linspace(0, x_max, num=n_bins, endpoint=True)
        p_hat, _ = np.histogram(x, bins=x_bins, density=True)
        p_hat /= p_hat.sum()
        x_ax = 0.5*(x_bins[:-1] + x_bins[1:]) # correct n_bins+1 issue
    if method == 'kernel':
        # gaussian, tophat, epanechnikov, exponential, linear, cosine
        kde = KernelDensity(bandwidth=0.25, kernel='epanechnikov')
        kde.fit(x[:,None])
        x_ax = np.linspace(0, x_max, num=n_bins, endpoint=True)
        log_p = kde.score_samples(x_ax[:, None])
        p_hat = np.exp(log_p)
        p_hat /= p_hat.sum()
    return x_ax, p_hat


def poincare_plot(I, I_min=None, I_max=None, doplot=False):
    """
    Construct Poincaré plot, get principal axes of main cloud
    Arguments:
        I: list of interval lengths (durations of respiratory cycles)
    Returns:
        sd1, sd2: length of principal axes
    """
    #print("\n[+] Poincare plot")
    if not I_min: I_min = I.min()
    if not I_max: I_max = I.max()
    #I_min, I_max = I.min(), I.max()
    #I_min, I_max = 0, 200
    I = I[I >= I_min]
    I = I[I <= I_max]
    nI = len(I)
    print("[+] Poincare plot function:")
    print(f"mean: {I.mean():.2f}, std: {I.std():.2f}")
    x, y = I[:-1], I[1:]
    data = np.vstack((x,y))
    x_m = x.mean()
    y_m = y.mean()
    #print(f"\tnumerical means: {x_m:.3f}, {y_m:.3f}")
    C = np.cov(data)
    #print(f"\tcov. matrix C: \n", C)
    #print("\tcov. matrix C: ", f"\n\t\t{C[0,0]:.3f}, {C[0,1]:.3f}", \
    #f"\n\t\t{C[1,0]:.3f}, {C[1,1]:.3f}")

    # method 1: manual PCA (eigen-analysis of data cov matrix)
    #print("\nMethod-1: manual PCA (diagon. cov. matrix)")
    L, V = np.linalg.eig(C)
    l0, l1 = L[0], L[1]
    v0, v1 = V[:,0], V[:,1]
    del L, V
    #print(f"\tC eigenvalues: l0 = {l0:.3f}, l1 = {l1:.3f}")
    #print("\tC eigenvectors: ")
    #print("\tv0 = ", np.round(v0,3))
    #print("\tv1 = ", np.round(v1,3))
    #print("\tcheck orthogonality: <v0, v1> = ", np.dot(v0, v1))

    # order eigenvalue magnitudes (small, large)
    #if (l0 > l1):
    #    l0, l1 = l1, l0
    #    v0, v1 = v1, v0

    # test vectors
    v_diag = np.array([1,1])/np.sqrt(2) # along main diagonal (identity, x=y)
    v_codiag = np.array([-1,1])/np.sqrt(2) # perpendicular to diagonal
    # test eigenvector directions
    ls_ordered = [None,None] # ordered eigenvalues
    vs_ordered = [None,None] # ordered eigenvectors
    t0_diag = np.abs(np.dot(v0, v_diag))
    t0_codiag = np.abs(np.dot(v0, v_codiag))
    #print(f"t0_diag={t0_diag:.3f}, t0_codiag={t0_codiag:.3f}")
    t1_diag = np.abs(np.dot(v1, v_diag))
    t1_codiag = np.abs(np.dot(v1, v_codiag))
    #print(f"t1_diag={t1_diag:.3f}, t1_codiag={t1_codiag:.3f}")
    eps = 5e-2
    if np.allclose(t0_diag,1,atol=eps):
        #print("v0 diagonal")
        vs_ordered[0] = v0
        ls_ordered[0] = l0
    if np.allclose(t0_codiag,1,atol=eps):
        #print("v0 co-diagonal")
        vs_ordered[1] = v0
        ls_ordered[1] = l0
    if np.allclose(t1_diag,1,atol=eps):
        #print("v1 diagonal")
        vs_ordered[0] = v1
        ls_ordered[0] = l1
    if np.allclose(t1_codiag,1,atol=eps):
        #print("v1 co-diagonal")
        vs_ordered[1] = v1
        ls_ordered[1] = l1

    #print(f"eigenvalues: {l0:.2e} {l1:.2e}")
    #print(f"eigenvectors: ({v0[0]:.3f}, {v0[1]:.3f}), ({v1[0]:.3f}, {v1[1]:.3f})")

    l0, l1 = ls_ordered
    v0, v1 = vs_ordered[0], vs_ordered[1]
    #print(f"ordered eigenvalues: {l0:.2e} {l1:.2e}")
    #print(f"ordered eigenvectors: ({v0[0]:.3f}, {v0[1]:.3f}), ({v1[0]:.3f}, {v1[1]:.3f})")

    s0_hat = np.sqrt(l0)
    s1_hat = np.sqrt(l1)
    #print(f"\tsqrt of eigenvalues: {s0_hat:.3f}, {s1_hat:.3f}")

    if doplot:
        plt.figure(figsize=(6,6))
        plt.plot(I[:-1], I[1:], 'ok', ms=6, alpha=0.5)
        plt.plot(I, I, '-k', lw=2)
        plt.plot(x_m, y_m, 'or', ms=8)
        plt.plot([x_m, x_m+s0_hat*v0[0]], [y_m, y_m+s0_hat*v0[1]], '-b', lw=5)
        plt.plot([x_m, x_m+s1_hat*v1[0]], [y_m, y_m+s1_hat*v1[1]], '-b', lw=5)
        #plt.xlim(I_min, I_max)
        #plt.ylim(I_min, I_max)
        plt.xlabel(r"$I_{n}$ [s]", fontsize=14)
        plt.ylabel(r"$I_{n+1}$ [s]", fontsize=14)
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    return s0_hat, s1_hat, v0, v1


def main():
    pass


if __name__ == "__main__":
    os.system("clear")
    main()
