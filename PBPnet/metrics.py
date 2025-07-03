import scipy
import torch
import numpy as np
import pandas as pd
from metrics_utils import *
import utility_loader
from tqdm import tqdm
from bpnetlite.performance import pearson_corr
from custom_bpnet import customBPNet
from tangermeme.ersatz import shuffle, dinucleotide_shuffle, randomize
import sklearn.metrics as skm

TFs_LIST=['CBX7', 'KDM2B', 'MTF2', 'PCGF1', 'PCGF2', 'PCGF6', 'RING1B', 'RYBP', 'SUZ12']

def tf_counts_corr(model, test_data, task, fold):
	"""For each TFs, compute the spearman correlation after selecting the appropriate predictions
	"""
	# Predict and return the whole tensors for predicted and observed counts
	# Fixing here single_metric=True as this is specific to single TF corr
	bpnet_pred_obs = model.predict(
		test_data,batch_size=256,
		shuffle_sequence=False, random_state=0,
		single_metric_counts=True, verbose=False
	)

	# Predict for shuffled version
	shuffle_pred_obs = model.predict(
		test_data,batch_size=256,
		shuffle_sequence=True, random_state=0,
		single_metric_counts=True, verbose=False
	)

	# Compute correlation
	df_corr = _pearson_corr(bpnet_pred_obs, shuffle_pred_obs, task, fold)
	return df_corr


def _pearson_corr(bpnet_pred_obs, shuffle_pred_obs, task, fold):
	
	# Given the pred and obs, compute correlation
	corr_bpnet, corr_shuffle, input_corr = [], [], []
	# extract counts for both predictions (BPNet and random)
	# calcualate corr
	corr_bpnet.append(pearson_corr(bpnet_pred_obs[0], bpnet_pred_obs[1]).numpy())
	input_corr.append(pearson_corr(bpnet_pred_obs[2], bpnet_pred_obs[1]).numpy())
	corr_shuffle.append(pearson_corr(shuffle_pred_obs[0], shuffle_pred_obs[1]).numpy())

	df_correlation = pd.DataFrame({
		'method': ['BPNet', 'Input', 'Shuffle'],
		'fold' : f'fold_{fold}',
		'task': task,
		'pearson': corr_bpnet + input_corr + corr_shuffle
	})

	# df_correlation = pd.DataFrame({
		# 'method': [item for sublist in [['BPNet']*len(tfs_name), ['Shuffle']*len(tfs_name)] for item in sublist],
		# 'task': [item for sublist in [tfs_name, tfs_name] for item in sublist],
		# 'pearson': [item for sublist in [corr_bpnet, corr_shuffle] for item in sublist]
	# })

	# Fix corr type
	df_correlation['pearson'] = df_correlation['pearson'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
	return df_correlation


# This a collection of metrics function from basepair models and other sources. I've added small tweaks
# Main authors: https://github.com/kundajelab/bpnet-manuscript/blob/master/basepair/metrics.py

def tf_profile_auprc(model, test_data, annot_regions):
	"""For each TFs, compute the auPRC at different binsizes after selecting the appropriate predictions and tracks
	"""

	y_true, y_pred = model.predict(
	test_data,batch_size=1024,
	shuffle_sequence=False, random_state=2,
	single_metric_counts=False, verbose=False, single_metric_profile=True
	)

	# Reshaping profiles to match the expected shape 

	reshaped_true = y_true.reshape(-1, 1000, 22).numpy()
	reshaped_pred = y_pred.reshape(-1, 1000, 22).numpy()
	
	# Instantiate a new df
	auprc_df = pd.DataFrame()
	counter_pos = 0
	
	for tf in tqdm(TFs_LIST):

		idx = annot_regions[annot_regions['name'] == tf].index

		# Observed vs predicted
		O_P_df = _eval_profile(reshaped_true[idx, :, counter_pos:counter_pos+2], reshaped_pred[idx, :, counter_pos:counter_pos+2])
		O_P_df['type'] = 'obs_vs_pred'

		# Observed vs shuffled
		O_S_df = O_P_df.copy()
		O_S_df['type'] = "obs_vs_random"
		O_S_df['auprc'] = O_P_df['random_auprc']

		df_final = pd.concat([O_S_df, O_P_df])
		df_final['task'] = tf
		auprc_df = pd.concat([auprc_df, df_final])
		counter_pos += 2
	
	return auprc_df


def _eval_profile(yt, yp,
				 pos_min_threshold=0.015,
				 neg_max_threshold=0.005,
				 required_min_pos_counts=2.5,
				 binsizes=[1, 2, 4, 10, 20, 50, 100]): #1, 2, 4, 10
	"""
	Evaluate the profile in terms of auPR

	Args:
	  yt: true profile (counts)
	  yp: predicted profile (fractions)
	  pos_min_threshold: fraction threshold above which the position is
		 considered to be a positive
	  neg_max_threshold: fraction threshold bellow which the position is
		 considered to be a negative
	  required_min_pos_counts: smallest number of reads the peak should be
		 supported by. All regions where 0.05 of the total reads would be
		 less than required_min_pos_counts are excluded
	"""
	# The filtering
	# criterion assures that each position in the positive class is
	# supported by at least required_min_pos_counts  of reads
	do_eval = yt.sum(axis=1).mean(axis=1) > required_min_pos_counts / pos_min_threshold

	# make sure everything sums to one
	yp = yp / yp.sum(axis=1, keepdims=True)
	fracs = yt / yt.sum(axis=1, keepdims=True)

	yp_random = permute_array(permute_array(yp[do_eval], axis=1), axis=0)
	out = []
	for binsize in binsizes:
		is_peak = (fracs >= pos_min_threshold).astype(float)
		ambigous = (fracs < pos_min_threshold) & (fracs >= neg_max_threshold)
		is_peak[ambigous] = -1
		y_true = np.ravel(bin_counts_amb(is_peak[do_eval], binsize))

		imbalance = np.sum(y_true == 1) / np.sum(y_true >= 0)
		n_positives = np.sum(y_true == 1)
		n_ambigous = np.sum(y_true == -1)
		frac_ambigous = n_ambigous / y_true.size

		# TODO - I used to have bin_counts_max over here instead of bin_counts_sum
		try:
			res = auprc(y_true,
						np.ravel(bin_counts_max(yp[do_eval], binsize)))
			res_random = auprc(y_true,
							   np.ravel(bin_counts_max(yp_random, binsize)))
		except Exception:
			res = np.nan
			res_random = np.nan

		out.append({"binsize": binsize,
					"auprc": res,
					"random_auprc": res_random,
					"n_positives": n_positives,
					"frac_ambigous": frac_ambigous,
					"imbalance": imbalance
					})

	return pd.DataFrame.from_dict(out)