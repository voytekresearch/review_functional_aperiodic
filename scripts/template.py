"""
Figure XXX: XXX



"""

# IMPORTS ######################################################################

# standard
import os
import numpy as np
import matplotlib.pyplot as plt

# custom
import sys
sys.path.append('code')
from settings import FIGURE_PATH, FIGURE_WIDTH, PANEL_FONTSIZE

# SETTINGS #####################################################################

# 

# figure
plt.style.use('mplstyle/trends_cogn_sci.mplstyle')

# MAIN ########################################################################

def main():

    # create output directory
    if not os.path.exists(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)


    print("FOOBAR")


if __name__ == "__main__":
    main()
