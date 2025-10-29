import os
# Ensure matplotlib uses a non-PGF backend before any imports that may load matplotlib
os.environ.setdefault('MPLBACKEND', 'Agg')

import _init_paths
import matplotlib
# Use a non-interactive backend that doesn't require TeX/PGF internals
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'uav123_10fps'

"""ostrack"""
trackers.extend(trackerlist(name='sglatrack', parameter_name='sglatrack', dataset_name=dataset_name,
                            run_ids=None, display_name='sglatrack_deit'))


dataset = get_dataset(dataset_name)

# Generate and save plots (PDF and optionally .tex if tikzplotlib is available)
plot_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

# Also print textual results
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

