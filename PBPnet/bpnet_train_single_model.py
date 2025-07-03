# Single python file to run training of BPNet model

import torch
from custom_bpnet import customBPNet
import os
import utility_loader
import h5py
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir='/data/mariani/specificity_bpnet/folds/'
task='SUZ12'

training_data = utility_loader.bpnet_DataLoader(
	hdf5_tensor=os.path.join(data_dir, '{}_5kb'.format(task), 'fold_0/train_data_fold_0.hdf5'), 
	in_window=6114, out_window=5000, random_state=0, num_workers=4, 
	batch_size=128, max_jitter=128, reverse_complement=True,shuffle=True
)

valid_data = utility_loader.bpnet_DataLoader(
	hdf5_tensor=os.path.join(data_dir, '{}_5kb'.format(task), 'fold_0/validation_data_fold_0.hdf5'), 
	in_window=6114, out_window=5000, random_state=0, num_workers=4, 
	batch_size=128, max_jitter=0, reverse_complement=False
)

name_model='{}_calibrated_5kb'.format(task)
outdir=f'/data/mariani/specificity_bpnet/single_models/{name_model}'
plot_metrics='figs'

if not os.path.exists(outdir):
	os.makedirs(outdir)


BPNet_dict = {
	"n_outputs":2,
	"n_control_tracks":2,
	"trimming":(6114-5000)//2,
	"alpha": 300, # Alpha RYBP = X, PCGF1/2 = 10, others 300
	"n_filters": 128,
	"n_layers":13,
	"name": os.path.join(outdir, name_model)
}

model = customBPNet(**BPNet_dict).cuda()

# Switch to AdamW?
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2) #PCGF1/6, KDM2B, RING1B, RYBP lr = 1e-3,
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, cooldown=2, patience=4)

# Model fitting
model.fit(
	training_data, 
	optimizer, 
	valid_data=valid_data, 
	valid_batch_size=1024, 
	max_epochs=500, 
	validation_iter=25,
	early_stopping=15*6 #This is not epochs but number of validations. I validated ~3/4 times per epoch
	)


if not os.path.exists(os.path.join(outdir, plot_metrics)):
	os.makedirs(os.path.join(outdir, plot_metrics))


log = pd.read_table(os.path.join(outdir, f'{name_model}.log'))


sns.lineplot(data=log, x="Epoch", y="Training MNLL", label="train")
sns.lineplot(data=log, x="Epoch", y="Validation MNLL", label="val")
plt.title("MNLL Loss")
plt.savefig(os.path.join(outdir, plot_metrics, 'MNLL_loss.pdf'))
plt.close()

sns.lineplot(data=log, x="Epoch", y="Training Count MSE", label="train")
sns.lineplot(data=log, x="Epoch", y="Validation Count MSE", label="val")
plt.title("Count MSE Loss")
plt.savefig(os.path.join(outdir, plot_metrics, 'MSE_loss.pdf'))
plt.close()

sns.lineplot(data=log, x="Epoch", y="Validation Profile Pearson", color="green", label="profile")
sns.lineplot(data=log, x="Epoch", y="Validation Count Pearson", color="brown", label="count")
plt.title("Validation Pearson")
plt.savefig(os.path.join(outdir, plot_metrics, 'pearson_validation.pdf'))
plt.close()

sns.lineplot(data=log, x="Epoch", y="valid_loss", color="purple", label="validation loss")
sns.lineplot(data=log, x="Epoch", y="best_loss", color="green", label="best_loss")
plt.title("validation loss")
plt.savefig(os.path.join(outdir, plot_metrics, 'validation_loss.pdf'))
