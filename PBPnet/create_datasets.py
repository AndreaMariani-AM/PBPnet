# This script creates the three tensors required to train the BPNet model.
# Tensors are sequences (B, 4, L), signals (B, len(signals), L) and controls (B, len(controls), L)
# Each triplet is derived for Train/Validation/Test split.
# I'll do a 3fold cross validation, testing three different folds.
# Fold 0, is the one that is fixed and chosen to have a split similar to affinity distillation paper (https://www.biorxiv.org/content/10.1101/2023.05.11.540401v1.full)
# Chrom 8/9 for testing and 16/17/18 for validation. As a rule of thumb the Validation and Testing should have at least ~10% of the regions
import torch

from tangermeme.io import extract_loci
import random
import os
import h5py
from dataset_utils import interleave_loci

# output directory
outDir="/hpcnfs/scratch/DP/amariani/Amariani/BPNet_data/output/folds"
folds="SUZ12_5kb"
if not os.path.exists(os.path.join(outDir, folds)):
	os.makedirs(os.path.join(outDir,  folds))

# Set training data info
# Let's start with the ones that have a motif and then add the rest
peaks = [
	# '/path/to/peak/files
     ]

# Genome
seqs = 'path/to/genome.fa' 

# Signal tracks
signals = [
	# '/path/to/pos_strand.bw', 
	# '/path/to/neg_strand.bw',

	]  # A set of stranded bigwigs

# Control tracks
controls =[
	# '/path/to/control/pos_strand.bw', 
	# '/path/to/control/neg_strand.bw',
	]
	 # A set of bigwigs

ignore_list = ignore=list('QWERYUIOPSDFHJKLZXVBNM')

random.seed(0)

print('This is the order of regions')
print(peaks)
print()
print('This is the order of signals')
print(signals)

# Creating the folds
for i in range(3):
	if not os.path.exists(os.path.join(outDir, folds, f'fold_{i}')):
		os.makedirs(os.path.join(outDir,  folds, f'fold_{i}'))
	# set the first fold to be fixed
	all_chroms = ['chr{}'.format(j) for j in range(1, 20)] + ['chrX']
	if i==0:
		valid_chroms = ['chr16', 'chr17', 'chr18']
		testing_chroms = ['chr8', 'chr9']
		training_chroms =[x for x in all_chroms if x not in valid_chroms+testing_chroms]
	elif i in [1,2]:
		random.seed(1 if i==1 else 42)
		# Random choice of splits based on a random state
		random_chroms = random.sample(training_chroms, 6) # 0:3 is validation, 3:6 is testing
		valid_chroms = random_chroms[0:3]
		testing_chroms = random_chroms[3:6]
		training_chroms =[x for x in all_chroms if x not in valid_chroms+testing_chroms]

	print(f'Extracting sequences for fold_{i}')
	print(f'Training chromosomes are: {training_chroms}')
	print(f'Validation chromosomes are: {valid_chroms}')
	print(f'Testing chromosomes are: {testing_chroms}')
	# Retrieve Trainig sequences sequences
	datasetsize = []

	for k in ['train', 'validation', 'test']:
		if k == 'train':
			chroms = training_chroms
			max_jitter = 128
		elif k == 'validation':
			chroms = valid_chroms
			max_jitter = 0
		else:
			chroms=testing_chroms
			max_jitter=0

		print(f'Retrieve {k} set ========================================================')
		seqs_array, signals_array, controls_array = extract_loci(
			peaks, seqs, signals, controls, chroms=chroms,
			in_window=6114, out_window=5000, max_jitter=max_jitter,
			verbose=True, ignore=ignore_list
			)

		if k == 'test':
			print(f'Retrieve {k} set order of regions ========================================================')
			loci = interleave_loci(peaks, chroms)
			loci.to_csv(os.path.join(outDir,  folds, f'fold_{i}', f'{k}_data_fold_{i}_regions.csv'), index=False, header=True)

		# save tensors shape
		datasetsize.append(seqs_array.shape[0])
		with h5py.File(os.path.join(outDir,  folds, f'fold_{i}', f'{k}_data_fold_{i}.hdf5'), 'w') as f:
			f.create_dataset('seqs', data=seqs_array.numpy())
			f.create_dataset('signal', data=signals_array.numpy())
			f.create_dataset('control', data=controls_array.numpy())
		
		del seqs_array, signals_array, controls_array # Does it save RAM?
	
	val_size = (datasetsize[1]*100 / sum(datasetsize))
	test_size = (datasetsize[2]*100 / sum(datasetsize))

	print(f'Validation size is: {val_size}%')
	print(f'Test size is: {test_size}%')
	print('===========================')
	print()
	