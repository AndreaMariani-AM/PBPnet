# Sightly modifies function from tangermeme.io from Jacob Schreiber 
# https://github.com/jmschrei/tangermeme/blob/main/tangermeme/io.py

import pandas as pd
import numpy as np
import re

def interleave_loci(loci, chroms):
	"""An internal function for processing the provided loci.

	There are two aspects that have to be considered when processing the loci.
	The first is that the user can pass in either strings containing filenames
	or pandas DataFrames. The second is that the user can pass in a single
	value or a list of values, and if a list of values the resulting dataframes
	must be interleaved.

	If a set of chromosomes is provided, each dataframe will be filtered to
	loci on those chromosomes before interleaving. If a more complicated form
	of filtering is desired, one should pre-filter the dataframes and pass those
	into this function for interleaving.


	Parameters
	----------
	loci: str, pandas.DataFrame, or list of those
		A filename to load, a pandas DataFrame in bed-format, or a list of
		either.

	chroms: list or None, optional
		A set of chromosomes to restrict the loci to. This is done before
		interleaving to ensure balance across sets of loci. If None, do not
		do filtering. Default is None.


	Returns
	-------
	interleaved_loci: pandas.DataFrame
		A single pandas DataFrame that interleaves rows from each of the
		provided examples.
	"""

	if chroms is not None:
		if not isinstance(chroms, (list, tuple)):
			raise ValueError("Provided chroms must be a list.")

	if isinstance(loci, (str, pd.DataFrame)):
		loci = [loci]
	elif not isinstance(loci, (list, tuple)):
		raise ValueError("Provided loci must be a string or pandas " +
			"DataFrame, or a list/tuple of those.")

	names = ['chrom', 'start', 'end']
	loci_dfs = []
	for i, df in enumerate(loci):
		string_name = df
		if isinstance(df, str):
			df = pd.read_csv(df, sep='\t', usecols=[0, 1, 2], 
				header=None, index_col=False, names=names)
		elif isinstance(df, pd.DataFrame):
			df = df.iloc[:, [0, 1, 2]].copy()
		else:
			raise ValueError("Provided loci must be a string or pandas " +
				"DataFrame, or a list/tuple of those.")

		if chroms is not None:
			df = df[np.isin(df['chrom'], chroms)]

		df['idx'] = np.arange(len(df)) * len(loci) + i
		name_df = re.search(r'e14-(\w+)_peaks\.bed', string_name).group(1)
		df['name'] = name_df
		loci_dfs.append(df)

	loci = pd.concat(loci_dfs)
	loci = loci.set_index("idx").sort_index().reset_index(drop=True)
	return loci
