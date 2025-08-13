"""
Figure 4:  Literature meta-analysis of publication related to aperiodic activity
and cognition

This figure presents a literature meta-analysis of publications related to 
aperiodic activity and its implications for cognitive processes. Cognitive
terms are imported from the Cognitive Atlas. LISC is used for text analysis.
"""

# IMPORTS ######################################################################

# standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize

from lisc import Counts, Words
from lisc.io import SCDB, load_object
from lisc.io import save_object
from lisc.data import Articles
from collections import Counter

# custom
import sys
sys.path.append('code')
from settings import FIGURE_PATH, FIGURE_WIDTH, PANEL_FONTSIZE

# SETTINGS #####################################################################

# Set to False to save runtime is counts analysis has been run
RUN_COUNTS_ANALYSIS = False
RUN_WORDS_ANALYSIS = False

# Search terms 
APERIODIC_TERMS = "aperiodic activity OR aperiodic exponent OR aperiodic offset OR aperiodic knee OR aperiodic timescale"
SELECT_TERMS = ['attention', 'memory', 'learning']

# figure
plt.style.use('mplstyle/trends_cogn_sci.mplstyle')

# SET-UP #######################################################################

# create output directory
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# MAIN ########################################################################

def main():

    # Run counts analysis or load results
    if RUN_COUNTS_ANALYSIS:
        print("Running counts analysis...")
        counts = run_count_analysis(APERIODIC_TERMS)
    else:
        print("Loading counts analysis results...")
        counts = load_object('counts_cogntiveatlas', SCDB("lisc_db"))

    # Run words analysis or load results
    if RUN_WORDS_ANALYSIS:
        print("Running words analysis...")
        run_words_analysis(APERIODIC_TERMS, SELECT_TERMS)
    else:
        print("Using existing words analysis results...")

    # create figure and gridspec
    print("Creating figure...")
    fig = plt.figure(figsize=[FIGURE_WIDTH, 3], constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=2, nrows=1, width_ratios=[1, 1])
    ax_b = plt.subplot(spec[1])

    # plot panels
    plot_counts(spec[0], counts)
    plot_words_analysis(ax_b, APERIODIC_TERMS, SELECT_TERMS)

    # add panel labels
    fig.text(0.01, 0.98, 'A.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.51, 0.98, 'B.', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save figure
    plt.savefig(os.path.join(FIGURE_PATH, 'figure_4'), bbox_inches='tight')


def run_count_analysis(aperiodic_terms):
    """
    Run counts analysis
    """

    # Load terms from CognitiveAtlas
    concepts_df = pd.read_csv("data/cogntiveatlas_terms/cognitive_atlas_concepts_unique.csv")
    cognitive_terms = concepts_df['name'].values

    # Set terms lists, indicating they are separate lists with the 'A' and 'B' labels
    counts = Counts()
    counts.add_terms(aperiodic_terms, dim='A')
    counts.add_terms(cognitive_terms, dim='B')

    # Collect co-occurrence data
    counts.run_collection()

    # Save out the counts object
    save_object(counts, 'counts_cogntiveatlas', directory=SCDB('lisc_db'))

    return counts


def plot_counts(subplot_spec, counts):
    """
    Plot co-occurrence counts for top terms
    """

    # get top terms
    n_to_plot = 20
    array_of_scores = counts.counts[0]
    list_of_terms = counts.terms['B'].terms
    sorted_indices = array_of_scores.argsort()[::-1]
    top_terms = []
    top_scores = []
    for i in range(0, n_to_plot, 10):
        top_indices = sorted_indices[i:i+10]
        top_terms.append([list_of_terms[j][0] for j in top_indices])
        top_scores.append(array_of_scores[top_indices])

    # replace "behavioral inhibition" with "behav. inhib."
    for terms in top_terms:
        terms[:] = [term.replace("behavioral inhibition (cognitive)", "behav. inhib.") for term in terms]

    # set shared colorbar limits
    cbar_limits = (0, max(array_of_scores))
    norm = Normalize(vmin=cbar_limits[0], vmax=cbar_limits[1])
    cmap = plt.get_cmap('viridis')

    # create nested gridspec
    spec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=subplot_spec)
    ax0 = plt.subplot(spec[0])
    ax1 = plt.subplot(spec[1])

    # # plot
    for ax, scores, terms in zip((ax0, ax1), top_scores, top_terms):
        ax.imshow(scores[:, None], cmap=cmap, norm=norm)
        ax.set_yticks(range(len(terms)), labels=terms)

    for ax in (ax0, ax1):
        ax.set_xticks([])

    # add text annotations for scores
    for i, scores in enumerate(top_scores):
        for j, score in enumerate(scores):
            ax = [ax0, ax1][i]
            ax.text(0, j, f"{score}", ha='center', va='center', color='black')


def run_words_analysis(aperiodic_terms, select_terms):
    # run a collection of words for the top terms

    # Create Words object
    words = Words()
    term_list = []
    for term in select_terms:
        term_list.append(f"{aperiodic_terms} AND {term}")
    words.add_terms(term_list)

    # Collect words data
    db = SCDB('../lisc_db')
    words.run_collection(usehistory=True, retmax=45, save_and_clear=True, 
                         directory=db)

    # Save out the words data
    save_object(words, 'top_term_words', directory=db)


def plot_words_analysis(ax, aperiodic_terms, select_terms):
    """
    plot cumulative counts for top terms
    """

    # get years for all top terms
    years = {}
    for term in select_terms:
        arts = Articles(f"{aperiodic_terms} AND {term}")
        arts.load(SCDB('lisc_db'))
        years_term = [art['year'] for art in arts]
        years[term] = years_term

    # compute cumulative counts and plot
    for term, year_list in years.items():
        year_counts = Counter(year_list)
        years_sorted = sorted(year_counts)
        counts_sorted = [year_counts[y] for y in years_sorted]
        cumulative_counts = [sum(counts_sorted[:i+1]) for i in range(len(counts_sorted))]
        years[term] = (years_sorted, cumulative_counts)
        ax.plot(years_sorted, counts_sorted, label=term, alpha=0.7)

    # # Plot line plot of cumulative counts for each term
    # for term, (years_sorted, cumulative_counts) in years.items():
    #     ax.plot(years_sorted, cumulative_counts, label=term)

    # label
    ax.set_xlabel('year')
    ax.set_ylabel('count')
    ax.set_title('Cumulative publications')
    # ax.set_yscale('log')  # Use logarithmic scale for better visibility
    ax.legend()


if __name__ == "__main__":
    main()
