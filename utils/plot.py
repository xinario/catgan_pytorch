import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import os

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})



def plot(name, value, iter):
	_since_last_flush[name][iter] = value

def flush(save_dir):
	prints = []

	for name, vals in _since_last_flush.items():
		_since_beginning[name].update(vals)

		x_vals = np.sort(_since_beginning[name].keys())
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig(os.path.join(save_dir, name.replace(' ', '_')+'.jpg'))

	_since_last_flush.clear()

