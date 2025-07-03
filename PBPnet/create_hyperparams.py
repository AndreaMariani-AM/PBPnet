# This python file creates different pickle files, containing hyperparams for different variations of the architecture
# As BPNet model is quite stable a full on grid search isn't necessary, and therefore i'll asses parameters independently

import os
import pickle


outDir="/data/mariani/specificity_bpnet/config_dict"
nameDir='/data/mariani/specificity_bpnet/output/models_hyperparams'
if not os.path.exists(outDir):
	os.makedirs(outDir)


# Base dictionary to be edited
base_dict = {
	"n_outputs":22,
	"n_control_tracks":2,
	"trimming":(2114-1000)//2,
	"alpha": 1,
	"n_filters": 64,
	"n_layers":8,
	"lr": 0.0004,
	"name": "placeholder"
}

dict_edits = {
	"alpha": [10, 100, 1000], #[1, 10, 100, 1000],
	"n_filters": [16, 32, 64], #[64, 128, 256] this was the original idea. It seems that only 64 is doing something, so maybe lowering it makes sense
	"n_layers": [7, 9, 11], #[7, 9, 11]
	"lr": [0.01, 0.004, 0.001, 0.0004], #[0.01, 0.004, 0.001, 0.0004]
}

# Update base_dict with different params
for k in dict_edits.keys():
	hyper_list = dict_edits[k]
	for v in hyper_list:
		tmp_dict = base_dict.copy()
		tmp_dict[k] = v
		tmp_dict["name"] = os.path.join(nameDir, f'model_{k}_{v}', f'model_{k}_{v}')
		print(tmp_dict)
		with open (os.path.join(outDir, f'train_dict_{k}_{v}.pkl'), 'wb') as f:
			pickle.dump(tmp_dict, f)