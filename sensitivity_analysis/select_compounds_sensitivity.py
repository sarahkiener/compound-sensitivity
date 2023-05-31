#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Find the Most Frequent Translations for a Compound
==================================================
Among the list of all translation hypotheses for
a given compound, find the most frequent candidates.

"""



import argparse
from pathlib import Path


def parse_args():
	parser = argparse.ArgumentParser(description="find most frequent candidates for a compound")
	parser.add_argument('-n', '--sent_id', type=str, required=True, help="ID of the analysed sentence")
	parser.add_argument('-t', '--target', type=str, required=True, help="target word")
	return parser.parse_args()



def find_most_freq(path):
	'''
	For a given German compound, find the candidate translations
	that occur most frequently on the the candidate pool.

	Args:
		path: 	Path to file with all translation hypotheses
				for a given compound
	'''

	cand_dict = {}

	with open(path, 'r', encoding='utf-8') as cand_file:
		for line in cand_file:
			line = line.strip()
			if line in cand_dict:
				cand_dict[line] += 1
			else:
				cand_dict[line] = 1

	sorted_cands = {k: v for k, v in sorted(cand_dict.items(), key=lambda item: item[1], reverse=True)}
	print('different words:', len(sorted_cands))
	for cand, count in sorted_cands.items():
		print(cand, count)



def main(args):
	path = Path(f'compound_candidates/{args.sent_id}_{args.target}.txt')
	find_most_freq(path)


if __name__ == '__main__':
	args = parse_args()
	main(args)
