#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Extract Unknown Words From MT Output
====================================
Identify unknown words by comparing each word of the MT output
to the vocabulary of the training data, the references and the 
source segments. If a word is not found in this vocabulary,
it is defined as unknown.

"""

from typing import BinaryIO, List
from pathlib import Path
import argparse
import string
import glob
import os


def parse_args():
	parser = argparse.ArgumentParser(description="Identify non-sensical compounds in translations.")
	parser.add_argument('-td', '--transdir', type=str, required=False, default='en-de', help="translation direction, e.g. en-de, de-en")
	parser.add_argument('-tf', '--transfile', type=str, required=True, help="path to file containing the translation output of the model")
	parser.add_argument('-o', '--outfile', type=str, required=True, help="path to output file to save the list of unknown words")
	return parser.parse_args()

def create_vocab_dict(infile: BinaryIO, vocab_dict: dict) -> dict:
	'''
	Create a vocabulary with words occuring in the training corpus
	or the reference translation.
	
	Args:
		infile:		Corpus file from which the words are extracted
		vocab_dict:	Vocabulary collected in previous runs of the loop

	Return:
		vocab_dict: Extracted vocabulary
	'''

	# Remove punctuation characters from the corpus and add each word to the vocab
	# (hyphens are kepts as they often form part of compounds)
	for line in infile:
		line = line.translate(str.maketrans('', '', '!"#$%()*+,./:;<=>?[]^_`{|}~„“'))
		words = line.strip().split()
		for word in words:
			word = word.strip()
			if word not in vocab_dict:
				vocab_dict[word] = 1
	return vocab_dict

def compare_translations(vocab_dict: dict) -> List[set]:
	'''
	Compare the words in the MT output to the vocabulary and the 
	corresponding source sentence to identify unknown words.

	Args:
		vocab_dict:	Vocabulary

	Return:
		List[set]:	List with unique unknown words found in MT output
	'''
	
	strange_words = []
	# Read in the MT output and the source sentences
	with open(args.transfile, 'r', encoding='utf-8') as trans_file, open(Path(f'../prepared_data/{args.transdir}.src'), 'r', encoding='utf-8') as src_file:
		for line1, line2 in zip(trans_file, src_file):
			# Remove punctutaion charachters (hyphens are kepts as they often form part of compounds)
			line1 = line1.translate(str.maketrans('', '', '!"#$%()*+,./:;<=>?[]^_`{|}~„“'))
			line2 = line2.translate(str.maketrans('', '', '!"#$%()*+,./:;<=>?[]^_`{|}~„“'))
			trans_words = line1.strip().split()
			src_words = line2.strip().split()
			# Compare the words in the MT output to the vocabulary and the corresponding source sentence
			# If the word is neither found in the vocab nor in the source, append it to the list of 
			# unknown words
			for word in trans_words:
				if word not in vocab_dict and word not in src_words:
					strange_words.append((word, counter))
	# deduplicate the list of unknown words
	return list(set(strange_words))

def main(args):
	# Create a vocabulary
	vocab = {}
	# Read in the reference translations and add words to the vocabulary
	with open(Path(f'../prepared_data/{args.transdir}.refs'), 'r', encoding='utf-8') as ref_file:
		print('Reading', args.transdir + '.refs')
		vocab = create_vocab_dict(ref_file, vocab)
		print('Entries in vocab:', len(vocab))

	# Read in the training corpus and add words to the vocabulary
	os.chdir('../wmt-2018_data')	
	for file in glob.glob('*.' + args.transdir[-2:]):
		with open(file, 'r', encoding='utf-8') as train_file:
			print('Reading', file)
			vocab = create_vocab_dict(train_file, vocab)
			print('Entries in vocab:', len(vocab))

	# Extract unknown words from the MT output sentences
	strange_words = compare_translations(vocab)
	
	# Write the unknown words and their line number to the output file
	with open(args.outfile, 'w', encoding='utf-8') as outfile:
		for word, line in strange_words:
			outfile.write(word + '\t' + str(line) + '\n')
	# Print number of unknown words found in the MT output
	print('---------------------')
	print('number of strange words:', len(strange_words))
	

if __name__ == '__main__':
	args = parse_args()
	main(args)