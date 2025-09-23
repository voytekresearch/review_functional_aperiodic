"""
Figure 2: Spike synchrony + damped oscillators
===============================

Spike synchrony:
This figure will demonstrate the relationship between population spike synchrony 
and the aperiodic exponent of the field potential in biophysically-informed
neural simulations.

Neural firing is simulated using a two-process rate function (Kelly, 2010):
Each neuron's rate is simulated as the sum of an independent rate process and a 
shared population rate. Each neuron has a particular coupling-strength which 
determines how much the global rate influences its firing rate. The rate process 
is modelled as 
$m + W + OU * CS$
where $m$ is the mean firing rate of the neuron, $W$ is white-noise, $OU$ is 
the global rate fluctuations (modelled as an Ornstein–Uhlenbeck process), and 
$CS$ is the coupling-strength

Damped oscillators:
This figure will demonstrate the relationship between damped oscillators and
the aperiodic exponent of the field potential in simulated neural data.

"""

# IMPORTS ######################################################################

# standard
import os
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from neurodsp.spectral import compute_spectrum

# spike simulation and analysis
from elephant.spike_train_generation import NonStationaryPoissonProcess
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import correlation_coefficient
import neo
import quantities as pq

# custom
import sys
sys.path.append('code')
from sim_utils import (sim_ou_process, comp_exp, spiketimes_to_spiketrains, 
                       spikes_to_lfp)
from settings import FIGURE_PATH, FIGURE_WIDTH, PANEL_FONTSIZE

# SETTINGS #####################################################################

# simulation
FS = 1000 # sampling frequency (1/dt)
N_SECONDS = 5 # duration of simulation (sec) 
N_NEURONS = 1000 # number of neurons in population
np.random.seed(0)

# analysis
F_RANGE = [1, 100]

# figure
plt.style.use('mplstyle/trends_cogn_sci.mplstyle')
COLORS = ["#7570b3", "#3FAA96", "#F39943"]

# SET-UP #######################################################################

# set random seed
np.random.seed(37)

# create output directory
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# MAIN ########################################################################

def main():

    # run simulation for panels A-C
    spikes, time, lfp, freqs, spectra = simulate_three_levels()
    sync, exp, reg = correlate_synchrony_and_exponent()

    # init figure
    fig = plt.figure(figsize=[FIGURE_WIDTH, 8], constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=5, figure=fig, width_ratios=[1,1],
                             height_ratios=[1, 1, 1, 2, 3], wspace=0)

    # plot panels
    plot_abc(fig, spec, spikes, time, lfp, freqs, spectra, sync, exp, reg)
    plot_d(fig, spec)

    # add subplot labels
    fig.text(0.00, 0.99, '(A)', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.00, 0.54, '(B)', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.52, 0.54, '(C)', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.00, 0.27, '(D)', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save
    plt.savefig(os.path.join(FIGURE_PATH, 'figure_2'), bbox_inches='tight')


def simulate_2_process_lfp(coupled, coupling_strength, n_seconds=1, fs=1000,
                          n_neurons=1000, mu_spike_rate=10, var_spike_rate=10):
    """
    simulate LFP

    Parameters
    ----------
    coupled : bool
        indicates whether the spiking activity is coupled to the global 
        population rate process.
    coupling_strength : float
        coupling strength value; indicates how tightly the spike rate of 
        each neuron is coupled to the global population rate process.
    n_seconds : float, optional
        duration of simulation (sec). The default is 1.
    fs : float, optional
        sampling frequency (Hz). The default is 1000.
    n_neurons : int, optional
        number of neurons in the population. The default is 1000.
    mu_spike_rate : flaot, optional
        mean firing rate of each neuron in the population. The default is 10.
    var_spike_rate : float, optional
        firing rate variance. The default is 10.

    Returns
    -------
    poisson_spikes : list
        list of spike trains.
    pop_rate : int, array
        population firing rate vector.
    lfp : flaot, array
        simulated local field potential.
    time : float, array
        time-vector for simulated lfp.

    """

    # simulate random process (shared rate function across the population)
    ou, _ = sim_ou_process(n_seconds, fs, 0.05, mu=0, sigma=var_spike_rate)

    # simulate spikes from random process
    poisson_spikes = []
    for _ in range(n_neurons):
        rate_i = (mu_spike_rate + \
                 var_spike_rate*(np.random.rand(n_seconds*fs)-0.5)) + \
                 (ou * coupled * (coupling_strength*np.random.rand()))
        rate_i[rate_i<0] = 0
        rate = neo.AnalogSignal(rate_i, units=pq.Hz, sampling_rate=fs*pq.Hz)
        nspp = NonStationaryPoissonProcess(rate)
        poisson_spikes.append(nspp.generate_spiketrain(as_array=True))

    # convert spike times to spike trains
    _, spikes_array = spiketimes_to_spiketrains(poisson_spikes, fs, n_seconds, 
                                                return_matrix=True)
    pop_rate = np.sum(spikes_array, axis=0)

    # simulate LFP from spikes
    lfp, time = spikes_to_lfp(spikes_array, fs)
    
    return poisson_spikes, pop_rate, lfp, time


def simulate_three_levels():
    """
    Run the simulation with 3 synchrony levels (desynchronized, low synchrony, 
    and high synchrony). This will be used for plotting.
    """

    # settings
    coupled = [0, 1, 1]
    coupling_strength = [0, 0.5, 1]

    # simulate
    spikes = []
    lfp_list = []
    for cp, cs in zip(coupled, coupling_strength):
        spikes_i, _, lfp_i, time = simulate_2_process_lfp(cp, cs, fs=FS,
                                                          n_seconds=N_SECONDS, 
                                                          n_neurons=N_NEURONS)
        spikes.append(spikes_i)
        lfp_list.append(lfp_i)
    lfp = np.vstack(lfp_list)

    # compute psd
    _, temp = compute_spectrum(lfp[0], FS, f_range=F_RANGE)
    spectra = np.zeros([len(lfp), len(temp)])    
    for i_lfp, signal in enumerate(lfp):
        freqs, spectra[i_lfp] = compute_spectrum(signal, FS, f_range=F_RANGE)

    return spikes, time, lfp, freqs, spectra


def correlate_synchrony_and_exponent():
    """
    Repeat the simulation with varying coupling strengths, compute the
    corresponding exponents and spike correlations, and correlate them.
    """

    # settings
    coupling_strength = np.linspace(0,1,11)
    coupled = np.ones(len(coupling_strength))

    # simulate LFP
    pop_spikes_x = []
    lfps_x = []
    pop_rates_x = []
    for ii, (cp, cs) in enumerate(zip(coupled, coupling_strength)):
        print(f"Running simulation {ii+1}/{len(coupled)}")
        poisson_spikes, pop_rate, lfp, _ = simulate_2_process_lfp(cp, cs, fs=FS,
            n_seconds=N_SECONDS, n_neurons=N_NEURONS)
        pop_spikes_x.append(poisson_spikes)
        pop_rates_x.append(pop_rate)
        lfps_x.append(lfp)
        
    # compute exp
    exponent = comp_exp(np.vstack(lfps_x), FS)

    # compute correlation/covarience measures
    corr_mat = np.zeros([len(coupling_strength),N_NEURONS,N_NEURONS])
    synchrony = np.zeros(len(coupling_strength))

    for ii in range(len(coupling_strength)):
        spike_times = pop_spikes_x[ii]

        n_units = len(spike_times)
        spike_trains = []
        for i_unit in range(n_units):
            sp = neo.SpikeTrain(spike_times[i_unit]*pq.s, N_SECONDS*pq.s)
            spike_trains.append(sp)

        # compute corr
        binned_spikes = BinnedSpikeTrain(spike_trains, bin_size=10*pq.ms)
        corr_mat[ii] = correlation_coefficient(binned_spikes)
        synchrony[ii] = np.mean(corr_mat[ii])

    # compute linear regression
    reg = linregress(synchrony, exponent)

    return synchrony, exponent, reg


def plot_abc(fig, spec, spikes, time, lfp, freqs, spectra, sync, exp, reg):
    """
    Plot Figure 2 - panels A-D.

    a, spike raster and LFP for 3 populations with varying synchrony levels
    b, power spectral density (PSD) for the 3 populations
    c, correlation between synchrony and exponent
    """

    # create subplots
    ax0 = fig.add_subplot(spec[0, :])
    ax1 = fig.add_subplot(spec[1, :])
    ax2 = fig.add_subplot(spec[2, :])
    ax3 = fig.add_subplot(spec[3, 0])
    ax4 = fig.add_subplot(spec[3, 1])

    # plot spikes and lfp
    for ii, ax in enumerate([ax0,ax1,ax2]):
        # plot spikes
        ax.eventplot(spikes[ii], color='grey')
        ax.set(ylabel='neuron #')
        ax.set_xlim([0, N_SECONDS])

        # plot LFP
        axr = ax.twinx() 
        axr.plot(time[:len(lfp[0])], lfp[ii], linewidth=1, color=COLORS[ii])
        axr.set(ylabel='voltage (au)')

    # labels
    for ax in [ax0, ax1]:
        ax.set_xticks([])
    ax2.set(xlabel='time (s)', ylabel='neuron #')

    # plot PSD
    labels = ['desync.', 'low-sync.', 'high-sync.']
    for ii in range(len(spectra)):
        ax3.loglog(freqs, spectra[ii], color=COLORS[ii])
        ax3.set(xlabel='frequency (Hz)', ylabel='power (au)')
    ax3.legend(labels, loc='lower left')

    # plot regression results
    colors = [COLORS[0], 'k', 'k', 'k', 'k', COLORS[1], 'k', 'k', 'k', 'k', 
              COLORS[2]] # color those from earlier panels
    ax4.scatter(sync, exp, color=colors)
    ax4.set(xlabel='mean correlation', ylabel='exponent')
    model = [sync.min() * reg[0] + reg[1],
             sync.max() * reg[0] + reg[1]]
    ax4.plot([sync.min(), sync.max()], model, color='k', linestyle='--',
             label=f'r: {np.round(reg[2], 3)} \np: {reg[3]:0.2e}')
    ax4.legend(loc='lower right')

    # set titles
    ax0.set_title('Desynchronized')
    ax1.set_title('Low-synchrony')
    ax2.set_title('High-synchrony')
    ax3.set_title('Power spectra')
    ax4.set_title('Linear regression')


def damped_oscillator(t, f, alpha, gamma):
    """Simulate a damped oscillator.
    
    Parameters
    ----------
    t : array_like
        Time array.
    f : float
        Frequency of the oscillator.
    alpha : float
        Amplitude scaling factor.
    gamma : float
        Damping factor.

    Returns
    -------
    array_like
        Damped oscillator.
    """

    return np.cos(2 * np.pi * f * alpha * t) * np.exp(-gamma * t)


def plot_d(fig, spec, n_seconds=10, fs=1000, osc_freq=10):
    """
    Plot Figure 2 - panel D.

    d, damped oscillators with varying damping factors and their spectra
    """

    # settings
    colors = sns.color_palette("viridis", 4)

    # simulate damped oscillator
    time = np.arange(0, n_seconds, 1/fs)
    damping_factors = [0.1, 1, 10, 50]
    signals = np.empty((len(damping_factors), len(time)))
    for ii, gamma in enumerate(damping_factors):
        signals[ii] = damped_oscillator(time, osc_freq, 1, gamma)

    # compute spectra
    freqs, spectra = compute_spectrum(signals, FS, method='welch', nperseg=FS*8, 
                                      f_range=[4, 100])

    # init nested gridspec
    gs = gridspec.GridSpecFromSubplotSpec(5, 3, subplot_spec=spec[4, :],
                                          width_ratios=[0.1, 2, 1],
                                          height_ratios=[0.1, 1, 1, 1, 1])
    axes_0 = [fig.add_subplot(gs[i, 1]) for i in range(1, 5)]
    ax1 = fig.add_subplot(gs[1:, 2])

    # plot damped signals
    for ii, (ax, signal_i) in enumerate(zip(axes_0, signals)):
        ax.plot(time, signal_i, color=colors[ii])
        ax.set(xlabel="time (s)")
        ax.set_xlim([0, 2])
        ax.label_outer()
        ax.set_ylim([-1, 1])
        ax.set_yticks([])
    for ax in axes_0[:-1]:
        ax.set_xticks([])
    axes_0[-1].set_xticks([0, 1, 2], labels=['0', '1', '2'])
    axes_0[0].set_title("Damped oscillators")
    fig.text(0.00, 0.14, 'amplitude (au)', va='center', rotation='vertical', 
             fontsize=10)

    # plot spectra
    for ii in range(spectra.shape[0]):
        ax1.loglog(freqs, spectra[ii], color=colors[ii],
                   label=f"\u03B3 = {damping_factors[ii]} Hz")
    ax1.set(xlabel="frequency (Hz)", ylabel="power (au)")
    ax1.set_xticks([10, 100], labels=['10', '100'])
    ax1.set_title("Power spectra")
    ax1.legend()


if __name__ == "__main__":
    main()
