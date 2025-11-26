#!/usr/bin/env python3

import os
import sys
import pandas as pd
import argparse
from glob import glob # pip install glob2

def merge_files(data_path, ProjectList):
	data_dir = os.path.abspath(data_path)
	if not os.path.exists(data_dir):
		print('* ERROR: could not locate data_dir=%s' % (data_dir))
		sys.exit(1)
	print(' ++ collapsing tablefiles across Projects=[%s], data_dir=%s'%(','.join(ProjectList),data_dir))
	## keep the specified input project order
	for iParc in ['aparc','aparc.a2009s','aparc.DKTatlas']:
		print(' ++ merging files for parcellation=%s'%(iParc))
		parc_files=[]
		for iProj in ProjectList:
			parc_files.extend(glob(os.path.join(data_dir, "%s_%s_merged_fs_stats.txt"%(iProj,iParc))))
		#print(' + parc_files =',parc_files)
		parc_df = pd.concat(pd.read_csv(f, sep = "\t") for f in parc_files)
		print(' ++ parc=%s, len(parc_files)=%d, parc_df.shape=[%dx%d]'%(iParc,len(parc_files),*parc_df.shape))
		out_file = os.path.join(data_dir,'%s_merged_fs_stats.txt'%(iParc))
		parc_df.to_csv(out_file,sep='\t')
		print('  + merged results for parc=%s saved to file=%s' % (iParc,out_file))
	return


class ArgumentParser(argparse.ArgumentParser):
	def __init__(self):
		super(ArgumentParser, self).__init__()
		self.description = '''Merge FreeSurfer Measure TableFiles across Projects by Parcellation.'''
		self.add_argument('-d', type=str, dest='data_path', required=True, help='Path to TableFiles (input and output)')
		self.add_argument('ProjectList', type=str, action='append', nargs='+', help='List of Projects to merge across [ABIDE HBN etc]')

if __name__ == '__main__':
	args = ArgumentParser().parse_args()
	#print(' + args.ProjectList =',*args.ProjectList)
	merge_files(args.data_path, *args.ProjectList)
