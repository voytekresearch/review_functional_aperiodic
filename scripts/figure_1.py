"""
Figure 1: Basic model of aperiodic activity

This figure will illustrate a basic model of aperiodic neural activity, 
i.e. the filtered point process model.

"""

# IMPORTS ######################################################################

# standard
import os
import numpy as np
from scipy.signal import detrend, get_window

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

from neurodsp.sim.transients import sim_synaptic_kernel
from neurodsp.utils import create_times
from neurodsp.spectral import compute_spectrum

# custom
import sys
sys.path.append('code')
from settings import FIGURE_PATH, FIGURE_WIDTH, PANEL_FONTSIZE
from sim_utils import sample_spikes, convolve_psps, get_spike_times

# SETTINGS #####################################################################

# LFP simulation
FS = 1000 # sampling frequency
N_SECONDS = 100 # duration of simulation (sec)
N_NEURONS = 100 # number of neurons in the population
MEAN_RATE = 10 # average rate of each neuron (Hz)

# figure
plt.style.use('mplstyle/trends_cogn_sci.mplstyle')

# SET-UP #######################################################################

# set random seed
np.random.seed(42)

# create output directory
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# MAIN ########################################################################

def main():
    """
    Plot Figure 1.

    a, synaptic current contributions to the LFP
    b, postsynaptic potentials time-series
    c, postsynaptic potentials spectra
    """
        
    # set up gridspec
    fig = plt.figure(figsize=[FIGURE_WIDTH, 6], constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, width_ratios=[1,1],
                             height_ratios=[1, 0.6, 1], hspace=0.1)
    ax_a0 = fig.add_subplot(spec[0, :])
    ax_a1 = fig.add_subplot(spec[1, :])
    ax_b = fig.add_subplot(spec[2, 0])
    ax_c = fig.add_subplot(spec[2, 1])

    # plot panels
    plot_panel_a(ax_a0, ax_a1)
    plot_panel_bc(fig, ax_b, ax_c)

    # add panel labels
    fig.text(0.01, 0.97, 'A.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.04, 0.35, 'B.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.56, 0.35, 'C.', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save
    plt.savefig(os.path.join(FIGURE_PATH, 'figure_1'), bbox_inches='tight')


def plot_panel_a(ax_a0, ax_a1):
    """
    Plot subpanel a, synaptic current contributions to the LFP
    """

    # settings
    n2plot = 10
    n_seconds = 0.5
    fs = 1000
    rate = 30
    tau_rise = 0.0001
    tau_decay = 0.002

    # run simulation
    results = sim_lfp(n_neurons=N_NEURONS, mean_rate=rate, n_seconds=n_seconds, 
                      fs=FS, tau_rise=tau_rise, tau_decay=tau_decay, 
                      return_spikes=True)
    lfp, time, spikes, _, _ = results

    # simulate a few currents for plotting
    temp, _ = convolve_psps(spikes[0], fs, tau_r=tau_rise, tau_d=tau_decay)
    currents = np.zeros((n2plot, len(temp)))
    for i_spiketrain in range(n2plot):
        current_i, time_current = convolve_psps(spikes[i_spiketrain], fs, 
                                                tau_r=tau_rise, tau_d=tau_decay)
        currents[i_spiketrain] = current_i

    # plot currents 
    for i_current in range(n2plot):
        ax_a0.plot(time_current, currents[i_current] + i_current, color='k')
    ax_a0.set(xlabel="", ylabel="current (au)")
    ax_a0.set_title("Synaptic currents")

    # add annotation to right of ax indication that this is a subset of currents
    ax_a0b = ax_a0.twinx()
    ax_a0b.set_ylim(ax_a0.get_ylim())
    ax_a0b.tick_params(axis='y', length=0)  # hide ticks
    ax_a0b.text(1.04, 0.5, "neuron index", transform=ax_a0.transAxes, 
            ha='center', va='center', color='k', rotation=270)
    ax_a0b.set_yticks(np.arange(0, n2plot))
    ax_a0b.set_yticklabels(['N', r'$\cdots$', '8', '7', '6', '5', '4', '3', '2', '1'])

    # plot LFP
    ax_a1.plot(time, lfp, color='k')
    ax_a1.set(xlabel="time (s)", ylabel="voltage (au)")
    ax_a1.set_title("Field potential")

    # remove y-ticks 
    for ax in [ax_a0, ax_a1]:
        ax.set_yticks([])


def plot_panel_bc(fig, ax_b, ax_c):
    """
    Plot subpanels.

    b, postsynaptic potentials time-series
    c, postsynaptic potentials spectra
    """

    # Figure A - add dirac

    # settings
    fs = 100000
    t_ker = 1
    timescales = np.logspace(-4.5, -3, 8)  # logspace from 10^-4 to 10^-3
    timescales_ms = timescales * 1000  # convert 
    nperseg = 2048

    # init figure and set colormap
    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.LogNorm(vmin=timescales_ms.min(), vmax=timescales_ms.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # simulate each
    temp, _ = sim_psp(fs, tau_r=1e-6, tau_d=0.002, t_ker=t_ker)
    psp = np.zeros((len(timescales), len(temp)))
    for ii, timescale in enumerate(timescales):
        psp[ii], time = sim_psp(fs, tau_r=0.0001, tau_d=timescale, t_ker=t_ker)

    # simulate a dirac delta function
    dirac_delta = np.zeros(int(fs * t_ker))
    dirac_delta[1] = np.max(psp)
    psp = np.vstack((psp, dirac_delta[None, :]))

    # apply hanning window, zero pad, and compute psd
    hanning_window = get_window('hann', psp.shape[1])
    psp_ = psp.copy()
    psp_ = psp_ * hanning_window[None, :]
    psp_ = np.pad(psp_, ((0, 0), (len(hanning_window)//2, 
                                  len(hanning_window)//2)), mode='constant')
    freqs, spectra = compute_spectrum(psp_, fs, nperseg=nperseg)
    freqs = freqs[2:-2]
    spectra = spectra[:, 2:-2]  # trim to remove edge effects from padding

    # plot
    for ii, timescale in enumerate(timescales_ms):
        ax_b.plot(time*1000, psp[ii], color=sm.to_rgba(timescale))
        ax_c.loglog(freqs, spectra[ii], color=sm.to_rgba(timescale))

    # plot delta function representation
    ax_b.plot([0, 0], [0, np.max(psp)], color='k')  # vertical line at t=0
    ax_b.plot(time*1000, np.zeros_like(time), color='k')  # horizontal line at y=0
    ax_c.loglog(freqs, spectra[-1], color='k')

    # label plot
    ax_b.set_title("Postsynaptic potential")
    ax_b.set(xlabel="time (ms)", ylabel="voltage (au)")
    ax_b.set_xlim([-0.05, 1.0])

    ax_c.set_title("Power spectrum")
    ax_c.set(xlabel="frequency (Hz)", ylabel="power (au)")

    # add colorbar for timescale
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cbax = inset_axes(ax_c, width="3%", height="90%", loc='lower left',
                      bbox_to_anchor=(1.02, 0.05, 1, 1), 
                      bbox_transform=ax_c.transAxes, borderpad=0)
    fig.colorbar(sm, cax=cbax, label='timescale (ms)')


def sim_lfp(n_neurons=1000, mean_rate=2, n_seconds=10, fs=1000, 
            tau_rise=0., tau_decay=0.01, t_kernel=None, return_spikes=False):
    """
    Simulate local field potentials (LFP) as filtered point processes.
    """

    # set kernel length and number of samples
    if t_kernel is None:
        t_kernel = 5 * tau_decay
    n_samples = int(fs * (n_seconds + t_kernel)) - 1

    # simulate spiking
    spikes = []
    for _ in range(n_neurons):
        rand_white = np.random.normal(loc=mean_rate, scale=mean_rate**0.5, 
                                      size=n_samples)
        spikes_i = sample_spikes(rand_white, fs)
        spikes.append(spikes_i)
    spikes = np.array(spikes)
        
    # get spike times
    spike_time_vector = (np.arange(0, n_samples) / fs) - (t_kernel / 2)
    spike_times = []
    for i_cell in range(len(spikes)):
        spike_times.append(get_spike_times(spikes[i_cell], spike_time_vector))

    # simulate LFP
    pop_spikes = np.sum(spikes, axis=0)
    lfp, time = convolve_psps(pop_spikes, fs, tau_r=tau_rise, tau_d=tau_decay,
                              t_ker=t_kernel)
    lfp = detrend(lfp, type='constant')

    if return_spikes:
        return lfp, time, spikes, spike_time_vector, spike_times 
    else:
        return lfp, time


def sim_psp(fs, tau_r=0., tau_d=0.01, t_ker=None):
    """
    simulate post-synaptic potential (PSP) kernel
    """
    
    # If not provided, compute t_ker as a function of decay time constant
    if t_ker is None:
        t_ker = 5. * tau_d

    # Simulate
    kernel = sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    n_seconds = len(kernel) / fs
    time = create_times(n_seconds, fs)

    return kernel, time


if __name__ == "__main__":
    main()
