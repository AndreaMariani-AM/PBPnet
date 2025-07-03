# These functions are taken from the package form gaugnerlab
# Orginal license: https://github.com/gagneurlab/concise/blob/master/LICENSE
import numpy as np
import sklearn.metrics as skm

MASK_VALUE = -1

def _mask_nan(y_true, y_pred):
	mask_array = ~np.isnan(y_true)
	if np.any(np.isnan(y_pred)):
		print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
			  format(np.sum(np.isnan(y_pred)), y_pred.size))
		mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
	return y_true[mask_array], y_pred[mask_array]


def _mask_value(y_true, y_pred, mask=MASK_VALUE):
	mask_array = y_true != mask
	return y_true[mask_array], y_pred[mask_array]


def _mask_value_nan(y_true, y_pred, mask=MASK_VALUE):
	y_true, y_pred = _mask_nan(y_true, y_pred)
	return _mask_value(y_true, y_pred, mask)

def auprc(y_true, y_pred):
	"""Area under the precision-recall curve
	"""
	y_true, y_pred = _mask_value_nan(y_true, y_pred)
	return skm.average_precision_score(y_true, y_pred)

def permute_array(arr, axis=0):
	"""Permute array along a certain axis

	Args:
	  arr: numpy array
	  axis: axis along which to permute the array
	"""
	if axis == 0:
		return np.random.permutation(arr)
	else:
		return np.random.permutation(arr.swapaxes(0, axis)).swapaxes(0, axis)
		
def bin_counts_max(x, binsize=2):
	"""Bin the counts
	"""
	if binsize == 1:
		return x
	assert len(x.shape) == 3
	outlen = x.shape[1] // binsize
	xout = np.zeros((x.shape[0], outlen, x.shape[2]))
	for i in range(outlen):
		xout[:, i, :] = x[:, (binsize * i):(binsize * (i + 1)), :].max(1)
	return xout


def bin_counts_amb(x, binsize=2):
	"""Bin the counts
	"""
	if binsize == 1:
		return x
	assert len(x.shape) == 3
	outlen = x.shape[1] // binsize
	xout = np.zeros((x.shape[0], outlen, x.shape[2])).astype(float)
	for i in range(outlen):
		iterval = x[:, (binsize * i):(binsize * (i + 1)), :]
		has_amb = np.any(iterval == -1, axis=1)
		has_peak = np.any(iterval == 1, axis=1)
		# if no peak and has_amb -> -1
		# if no peak and no has_amb -> 0
		# if peak -> 1
		xout[:, i, :] = (has_peak - (1 - has_peak) * has_amb).astype(float)
	return xout