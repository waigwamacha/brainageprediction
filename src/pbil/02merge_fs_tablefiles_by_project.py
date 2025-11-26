#!/usr/bin/env python3

import os
import sys
import pandas as pd
import argparse
from glob import glob # pip install glob2

def merge_files(data_path, project):
	data_dir = os.path.abspath(data_path)
	if not os.path.exists(data_dir):
		print('* ERROR: could not locate data_dir=%s' % (data_dir))
		sys.exit(1)
	print(' ++ collapsing site tablefiles for all parcellations within Project=%s, data_dir=%s'%(project,data_dir))
	## all parcellations share aseg_stats 
	aseg_files = glob(os.path.join(data_dir, "%s_*_aseg_stats.txt"%(project)))
	aseg_df = pd.concat(pd.read_csv(f, sep = "\t") for f in aseg_files)
	print(' - aseg_df.shape:', aseg_df.shape)
	## collapse across each Parcellation, combining each with aseg_stats
	for iParc in ['aparc','aparc.a2009s','aparc.DKTatlas']:
		parc_df = aseg_df.copy()
		for iMeas in ['area','thickness','thicknessstd','volume','meancurv','gauscurv','foldind','curvind']:
			for iHemi in ['lh','rh']:
				meas_files = glob(os.path.join(data_dir, "%s_*_%s_%s_%s_stats.txt"%(project,iParc,iMeas,iHemi)))
				if len(meas_files) < 1:
					continue
				meas_df = pd.concat(pd.read_csv(f, sep = "\t") for f in meas_files)
				print(' ++ project=%s,parc=%s,meas=%s,hemi=%s, len(meas_files)=%d, meas_df.shape=%d x %d'%(project,iParc,iMeas,iHemi,len(meas_files),*meas_df.shape))
				parc_df = pd.merge(parc_df,meas_df) #how='inner', on='Scan_ID')
				print(' -- parc_df.shape:',parc_df.shape)
		print('+ project=%s, parc=%s, parc_df.shape = '%(project,iParc),parc_df.shape)
		print(parc_df.head(5))
		out_file = os.path.join(data_dir,'%s_%s_merged_fs_stats.txt'%(project,iParc))
		parc_df.to_csv(out_file,sep='\t')
		print(' + results for project=%s,parc=%s saved to file = %s' % (project,iParc,out_file))

class ArgumentParser(argparse.ArgumentParser):
	def __init__(self):
		super(ArgumentParser, self).__init__()
		self.description = """Merge FreeSurfer Measure TableFiles across Sites by Parcellation and Project."""
		self.add_argument('-d', type=str, dest='data_path', required=True, help='Path to TableFiles (input and output)')
		self.add_argument('-p', type=str, dest='project', required=True, help='Project to merge')

if __name__ == "__main__":
	args = ArgumentParser().parse_args()	
	merge_files(args.data_path, args.project)
