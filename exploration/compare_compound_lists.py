#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Compare Compound Lists
=======================
To ensure that the lists of mistranslated compounds are consistent across models,
each list of unknown words is semi-automatically compared to all the other lists edited so far.
"""


import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="compare lists of compounds")
	parser.add_argument('-i', '--input', type=str, required=True, nargs='+', help="lists of mistranslated compounds to which the current compound list is compared")
	parser.add_argument('-c', '--compound-list', type=str, required=True, help="candidate list of compounds")
	parser.add_argument('-u', '--unknown-words', type=str, required=True, nargs='+', help="list of all unknown words")
	return parser.parse_args()



def compare_compound_lists():
	# Create a set of mistranslated compounds found in the 
	# output of the models analyzed so far
	compound_set = set()
	for file in args.input:
		with open(file, 'r', encoding='utf-8') as infile:
			for line in infile:
				compound = line.strip()
				compound_set.add(compound)

	# Create a set of unknown words that are not compounds and of compounds  
	# that are considered as correct translations. 
	# These words should not appear in the list of mistranslated compounds
	deleted_set = set()
	for file in args.unknown_words:
		with open(file, 'r', encoding='utf-8') as unknown_word_file:
			for line in unknown_word_file:
				word, sent_id = line.strip().split('\t')
				if word not in compound_set:
					deleted_set.add(word)

	# Compare the candidate list of unknown compounds to the set of 
	# mistranslated compounds identified so far and to the set of words 
	# that are either not compounds or are a correctly formed compound
	with open(args.compound_list, 'r', encoding='utf-8') as comp_list:
		for line in comp_list:
			compound = line.strip()
			if compound in deleted_set:
				print('deleted:', compound)	
			elif compound in compound_set:
				print('in list:', compound)
			else:
				print(compound)
	
	
def main(args):
	compare_compound_lists()


if __name__ == '__main__':
	args = parse_args()
	main(args)
