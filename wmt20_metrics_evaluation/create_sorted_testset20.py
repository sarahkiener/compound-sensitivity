#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Create the Sorted Test Set 
==========================
From the WMT 2020 Metrics Shared Task test set, 
extract the source, refernce and hypotheses for 
a given language pair and write them into the 
outfile in that order that is required to run the 
official WMT 2020 Metrics Shared Task evaluation script.

"""


import csv
import argparse
import fileinput
from pathlib import Path


def parse_args():
	parser = argparse.ArgumentParser(description="Create a sorted testset for WMT metrics scripts")
	parser.add_argument('-i', '--input', type=str, required=False, default='../wmt20/wmt20metrics/newstest2020/txt', help="directory with input files")
	parser.add_argument('-d', '--details', type=str, required=False, default='mt_ref_combis.tsv', help="file with reference and MT hypotheses combinations")
	parser.add_argument('-lp', '--lang_pair', type=str, required=False, default='en-de', help="language pair to analyse")
	parser.add_argument('-o', '--outfile', type=str, required=True, help="output file with sorted reference and MT hypo pairs")
	return parser.parse_args()


def create_sorted_testset():
	with open(args.outfile, 'wt', encoding='utf-8') as outfile:
		testset_writer = csv.writer(outfile, delimiter = ',')
		testset_writer.writerow(['lp', 'src', 'mt', 'ref'])

		# Retrieve the source and reference sentences for the given language pair
		src_path = Path(f'{args.input}/sources/newstest2020-{args.lang_pair[:2]}{args.lang_pair[3:]}-src.{args.lang_pair[:2]}.txt')
		ref_path = Path(f'{args.input}/references/newstest2020-{args.lang_pair[:2]}{args.lang_pair[3:]}-ref.{args.lang_pair[3:]}.txt')
		
		# Open the tsv-file with information on the language pair,
		# test set, reference set and system names. Extract
		# the information for the given language pair.
		with open(args.details, 'r', encoding='utf-8') as details_file:
			for line in details_file:
				line_list = line.strip().split('\t')
				lp = line_list[1]
				refset = line_list[3]
				sys_name = line_list[4]
				if lp == args.lang_pair and refset == 'newstest2020':
					file_name = refset + '.' + lp + '.' + sys_name + '.txt'
					mt_path = Path(f'{args.input}/system-outputs/{args.lang_pair}/{file_name}')
					
					# Write the source, reference and hypotheses into the outfile
					# in the order required to run the WMT 2020 Metrics Shared 
					# Task evaluation.
					with open(src_path, 'r', encoding='utf-8') as src_file, open(mt_path, 'r', encoding='utf-8') as mt_file, open(ref_path, 'r', encoding='utf-8') as ref_file:
						for src, mt, ref in zip(src_file, mt_file, ref_file):
							src = src.strip()
							mt = mt.strip()
							ref = ref.strip()
							testset_writer.writerow([args.lang_pair, src, mt, ref])
	


def main(args):
	create_sorted_testset()


if __name__ == '__main__':
	args = parse_args()
	main(args)