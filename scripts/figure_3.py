"""
Figure 3: Neural variability 

This figure will demonstrate the relationship between neural variabilty and the 
aperiodic exponent. Through simulation, it will be demonstrated that a decrease 
in the aperiodic exponent (i.e. a counter-clockwise rotation or "flattening" 
of the power spectra) within trials results in a reduction in 
trial-to-trial variabilty.

"""

# IMPORTS ######################################################################

# standard
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import linregress

from neurodsp.sim import sim_powerlaw
from neurodsp.sim.utils import rotate_timeseries
from neurodsp.spectral import compute_spectrum
from neurodsp.utils import create_times

# custom
import sys
sys.path.append("code")
from settings import FIGURE_PATH, FIGURE_WIDTH, PANEL_FONTSIZE

# settings #####################################################################

# simulation settings
FS = 1000 # sampling frequency (Hz)
N_SECONDS = 2 # signal duration (sec)
N_TRIALS = 100 # number of trials to simulate
EXPONENT = -1.5 # aperiodic exponent for first half of signal (should be negative)
DELTA_EXPONENT = -0.5 # change in aperiodic exponent (negative value for counter-clockwise rotation (i.e. flattening)
F_ROTATION = 40 # rotation frequency (Hz)

# figure settings
plt.style.use('mplstyle/trends_cogn_sci.mplstyle')
COLS = (np.array([60,171,147])/255,
        np.array([244,157,70])/255)

# SET-UP #######################################################################

# set random seed
np.random.seed(42)

# create output directory
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# MAIN #########################################################################

def main():

    # simulate neural time-series and compute spectra
    time, signal, freqs, spectra = simulate_time_series()

    # regress ttv on aperiodic exponents
    exponent, ttv, fit = run_simulation()

    # plot results
    plot_results(time, signal, freqs, spectra, exponent, ttv, fit)


def simulate_time_series():
    """
    Simulate a collection of neural time series with a sudden shift in the 
    broadband spectral profile; this is analogous to the task-associated 
    shifts in the aperioidic exponent observed in human intracranial EEG.
    See:
        * Preston et. al., J. Neurosci., 2025
        * Podvalney et. al., J. Neurophysiol., 2015
    """

    # define time vector
    time = create_times(N_SECONDS, FS, start_val=-N_SECONDS/2)

    # initialize array
    n_samples = N_SECONDS*FS//2
    signal = np.zeros([N_TRIALS, n_samples*2])

    # simulate neural time-series
    for i_trial in range(N_TRIALS):
        # simulate neural time-series (first half of trial)
        i_sig = sim_powerlaw(N_SECONDS/2, FS, exponent=EXPONENT, 
                             f_range=[1,None])
        signal[i_trial, :n_samples] = i_sig
        
        # rotate simulated neural time-series (second half of trial)
        signal[i_trial, n_samples:] = rotate_timeseries(i_sig, FS, 
                                                       DELTA_EXPONENT, 
                                                       F_ROTATION)

    # compute spectra for pre- and post-stimulus periods
    freqs, psd_0 = compute_spectrum(signal[:,:n_samples], fs=FS,
                                    method='medfilt', f_range=[2,200])
    _, psd_1 = compute_spectrum(signal[:,n_samples:], fs=FS,
                                method='medfilt', f_range=[2,200])
    spectra = np.vstack([np.median(psd_0, axis=0), np.median(psd_1, axis=0)])

    return time, signal, freqs, spectra


def run_simulation():
    """
    Simulate neural time-series data with varying aperiodic exponents and
    compute trial-to-trial variability (TTV).
    """

    # settings
    init_exponent = -3 # initial aperiodic exponent
    deltas = np.arange(-2, 0.25, 0.25) # degrees of flattening

    # initialize array
    n_samples = N_SECONDS*FS//2
    signals = np.zeros([len(deltas), N_TRIALS, n_samples])

    # simulate
    for i_trial in range(N_TRIALS):
        # simulate neural time-series
        signal_i = sim_powerlaw(N_SECONDS/2, FS, exponent=init_exponent)
        # signal = sim_powerlaw(N_SECONDS/2, FS, exponent=exponent, f_range=[1, None]) # same results
        
        # rotate
        for i_delta, delta in enumerate(deltas):
            signals[i_delta, i_trial] = rotate_timeseries(signal_i, FS, delta, 
                                                          F_ROTATION)

    # calculate TTV
    ttv = np.mean(np.std(signals, axis=1)**2, axis=1)

    # run regression
    exponent = init_exponent - deltas
    reg = linregress(np.log10(ttv), exponent)
    fit = reg[0] * np.log10(ttv) + reg[1]

    return exponent, ttv, fit


def plot_results(time, signal, freqs, spectra, exponent, ttv, fit):
    """
    Plot Figure 3.
    
    a, Simulated neural time-series for several example trials. 
    b, Average pre- and post-stimulus power spectral density (PSD).
    c, Trial-to-trial variability (TTV) across time. 
    d, TTV v. aperiodic exponent with linear regression results.
 
    """

    # init
    n_samples = N_SECONDS*FS//2

    # create figure and gridspec
    fig = plt.figure(figsize=[FIGURE_WIDTH, 4], constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=2, nrows=2, hspace=0.1,
                             width_ratios=[1, 0.36], height_ratios=[1,1])
    ax_b = fig.add_subplot(spec[0,1])
    ax_c = fig.add_subplot(spec[1,0])
    ax_d = fig.add_subplot(spec[1,1])

    # plot subplot a: Time-series
    spec_a = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=spec[0,0])
    ax_a0 = fig.add_subplot(spec_a[0])
    ax_a1 = fig.add_subplot(spec_a[1])
    ax_a2 = fig.add_subplot(spec_a[2])
    ax_a3 = fig.add_subplot(spec_a[3])
    ax_a4 = fig.add_subplot(spec_a[4])
    axes_a = [ax_a0, ax_a1, ax_a2, ax_a3, ax_a4]

    for ii, ax in enumerate(axes_a):
        ax.plot(time[:n_samples], signal[ii, :n_samples], color=COLS[0])
        ax.plot(time[n_samples:], signal[ii, n_samples:], color=COLS[1])
        for loc in ['top', 'bottom', 'right', 'left']:
            ax.spines[loc].set_visible(False)
        ax.set_yticks([])
        if ii != 4:
            ax.set_xticks([])
    ax_a4.spines['bottom'].set_visible(True)
    ax_a4.set(xlabel='time (s)')
    fig.text(0.015, 0.76, 'Voltage (au)', ha='center', va='center', rotation=90, 
             fontsize=8)

    # plot subplot b: Power spectra (pre v. post)
    ax_b.loglog(freqs, spectra[0], label='pre', color=COLS[0])
    ax_b.loglog(freqs, spectra[1], label='post', color=COLS[1])
    ax_b.set(xlabel='frequency (Hz)', ylabel='power (au)')
    ax_b.legend(loc='upper right')

    # plot subplot c: Trial-to-trial variability (TTV) v. time
    sig_var = np.std(signal, 0)**2 # compute across-trial variance
    sig_var_win = np.array([sig_var[:n_samples].mean(), 
                            sig_var[n_samples:].mean()])
    ax_c.plot(time[:n_samples], sig_var[:n_samples], color=COLS[0])
    ax_c.plot(time[n_samples:], sig_var[n_samples:], color=COLS[1])
    ax_c.plot(time[:n_samples], np.repeat(sig_var_win[0], n_samples), color='k')
    ax_c.plot(time[n_samples:], np.repeat(sig_var_win[1], n_samples), color='k')
    ax_c.axvline(0, color='grey', linestyle='--', linewidth=3)
    ax_c.set(xlabel='time (s)', ylabel='variance')

    # plot subplot d: TTV v. Exponent
    ax_d.plot(ttv, exponent, color='k', marker='o')
    ax_d.plot(ttv, fit, color='r')
    ax_d.set_xscale('log')
    ax_d.set(xlabel='variability (TTV)', ylabel='exponent')

    # set titles
    ax_a0.set_title('Simulated neural time-series')
    ax_b.set_title('Power spectra')
    ax_c.set_title('Trial-to-trial variability (TTV)')
    ax_d.set_title('TTV v. exponent')

    # limit x-ticks
    for ax in [ax_a4, ax_c]:
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([-1, 0, 1])

    # beautify
    for ax in [ax_b, ax_c, ax_d]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # add figure panel labels
    fig.text(0.002, 0.96, '(A)', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.700, 0.96, '(B)', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.002, 0.48, '(C)', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.700, 0.48, '(D)', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save/show
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_3'))


if __name__ == "__main__":
    main()
