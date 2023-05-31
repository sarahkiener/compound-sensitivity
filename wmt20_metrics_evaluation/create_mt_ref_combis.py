#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Create the Input File for the WMT 2020 Metrics Shared Task Evaluation
=====================================================================
Combine the information on the metric, language pair, test set, 
reference set, and MT system in a tsv-file.

"""


import argparse
from pathlib import Path


def parse_args():
	parser = argparse.ArgumentParser(description="Create file with MT-ref combis for a language pair")
	parser.add_argument('-i', '--infile', type=str, required=False, default="mt_ref_combis.tsv", help="file containing the hyposet - refset - MTsys combis for each language pair")
	parser.add_argument('-d', '--details', type=str, required=False, default="../wmt20/wmt20metrics/newstest2020/txt/details")
	parser.add_argument('-lp', '--lang_pair', type=str, required=False, default='en-de', help="language pair to analyse")
	parser.add_argument('-m', '--metric', type=str, required=True, help="name of metric to evaluate")
	parser.add_argument('-o', '--outfile', type=str, required=True, help="output file to save full info on each hypothesis")
	return parser.parse_args()


def create_mt_ref_combis():
	'''
	Create a tsv-file for each metric in the input format that is 
	required for the WMT 2020 Metrics Shard Task evaluation. 
	The created file contains the metric name, language pair, 
	test set, reference set (A, B or Paraphrase), MT system name,
	article ID, and sentence ID.
	'''

	# Combine the infos of the MT system - reference set combinations 
	# and the details of the test set for a given language pair (article ID and sentence ID) 
	details_path = Path(f'{args.details}/{args.lang_pair}.txt')
	with open(args.infile, 'r', encoding='utf-8') as infile, open(args.outfile, 'w', encoding='utf-8') as outfile:
		# Extract the language pair and the reference set
		for line in infile:
			line_list = line.strip().split('\t')
			lp = line_list[1]
			refset = line_list[3]

			# Iterate through the details and extract the article ID 
			# and the sentence ID
			if lp == args.lang_pair and refset == 'newstest2020':
				with open(details_path, 'r', encoding='utf-8') as details_file:
					for detail_line in details_file:
						detail_list = detail_line.strip().split('\t')
						article_id = detail_list[3]
						sent_id = detail_list[4]
						new_line = args.metric + '\t' + '\t'.join(line_list[1:]) + '\t' + article_id + '\t' + sent_id + '\n'
						outfile.write(new_line)


def main(args):
	create_mt_ref_combis()


if __name__ == '__main__':
	args = parse_args()
	main(args)