import torch
import h5py
from tangermeme.io import extract_loci
import numpy as np

# Slight different implementation of the dataloading process to accomodate large datasets. I'll create and save the Tensors correctly stacked
# and then wrap it up with a Dataset and the DataLoader
# Kudos to jmschrei:https://github.com/jmschrei/bpnet-lite/blob/master/bpnetlite/io.py

class customDataGenerator(torch.utils.data.Dataset):
	"""A data generator for BPNet inputs.

	This generator takes in an extracted set of sequences, output signals,
	and control signals, and will return a single element with random
	jitter and reverse-complement augmentation applied. Jitter is implemented
	efficiently by taking in data that is wider than the in/out windows by
	two times the maximum jitter and windows are extracted from that.
	Essentially, if an input window is 1000 and the maximum jitter is 128, one
	would pass in data with a length of 1256 and a length 1000 window would be
	extracted starting between position 0 and 256. This  generator must be 
	wrapped by a PyTorch generator object.

	Parameters
	----------
	sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		A one-hot encoded tensor of `n` example sequences, each of input 
		length `in_window`. See description above for connection with jitter.

	signals: torch.tensor, shape=(n, t, out_window+2*max_jitter)
		The signals to predict, usually counts, for `n` examples with
		`t` output tasks (usually 2 if stranded, 1 otherwise), each of 
		output length `out_window`. See description above for connection 
		with jitter.

	controls: torch.tensor, shape=(n, t, out_window+2*max_jitter) or None, optional
		The control signal to take as input, usually counts, for `n`
		examples with `t` strands and output length `out_window`. If
		None, does not return controls.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 0.

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is False.

	random_state: int or None, optional
		Whether to use a deterministic seed or not.
	"""

	def __init__(self, hdf5, in_window=2114, 
		out_window=1000, max_jitter=0, reverse_complement=False, 
		random_state=None):

		self.hdf5_path = hdf5
		self.in_window = in_window
		self.out_window = out_window
		self.max_jitter = max_jitter
		self.reverse_complement = reverse_complement
		self.random_state = np.random.RandomState(random_state)

		# Open HDF5 file in read mode and get dataset sizes
		with h5py.File(self.hdf5_path, 'r') as f:
			self.data_len = f['seqs'].shape[0]

	def __len__(self):
		return self.data_len

	def __getitem__(self, idx):
		with h5py.File(self.hdf5_path, 'r') as f:
			i = self.random_state.choice(self.data_len)
			j = 0 if self.max_jitter == 0 else self.random_state.randint(self.max_jitter * 2)
			
			X = torch.tensor(f['seqs'][i][:, j:j + self.in_window], dtype=torch.float32)
			y = torch.tensor(f['signal'][i][:, j:j + self.out_window], dtype=torch.float32)
			X_ctl = torch.tensor(f['control'][i][:, j:j + self.in_window], dtype=torch.float32)
			
			if self.reverse_complement and self.random_state.choice(2) == 1:
				X = torch.flip(X, [0, 1])
				y = torch.flip(y, [0, 1])
				X_ctl = torch.flip(X_ctl, [0, 1])

		return X, X_ctl, y

def bpnet_DataLoader(
	hdf5_tensor, in_window=2114, out_window=1000, max_jitter=0,
	reverse_complement=False, random_state=None, pin_memory=True,
	num_workers=0, batch_size=32, shuffle=False,
	):
	"""
	Custom Dataset and DataLoader based on BPNet-lite

	seqs, signals and controls are contained in the hdf5 tensor at these idxs:
	0 = seqs, 1 = signal, 2= control
	"""
	# DataLoader agnostic to training/validation split
	X_gen = customDataGenerator(
		hdf5=hdf5_tensor, in_window=in_window, out_window=out_window,
		max_jitter=max_jitter, reverse_complement=reverse_complement, random_state=random_state
	)

	X_gen = torch.utils.data.DataLoader(X_gen, pin_memory=pin_memory,
	num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)

	return X_gen