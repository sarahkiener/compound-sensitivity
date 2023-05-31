#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Create Validation Set
=====================
From the WMT 2020 Metrics Shared Task data set,
extract all MT hypotheses that correspond to 100
different source sentences as validation set.
In total, 722 segments are extracted as validation set.

NOTE:	The test set created in this script is NOT used
		for evaluation of the new metrics. Rather, the 
		en-de portion of the complete official test set 
		of the WMT 2020 Metrics Shared Task is used.
"""


import csv
from typing import BinaryIO


def dev_test_split(infile: BinaryIO, out_dev: str, out_test: str):
	"""
	From the en-de portion of the WMT 2020 Metrics Shared Task
	data set, extract all MT hypotheses that correspond to
	100 different source sentences. These 722 segments form
	the validation set. Write the other segments to the test set.

	Args:
		infile:		en-de portion of the WMT 2020 Metrics Shared
					Task data set with direct assessment scores
		out_dev:	validation set
		out_test:	test set 
	"""

	# Prepare the output csv-files for the dev and test sets
	with open(out_dev, 'wt', encoding='utf-8') as dev_file, open(out_test, 'wt', encoding='utf-8') as test_file:
		dev_writer = csv.writer(dev_file, delimiter=',')
		dev_writer.writerow(['lp', 'src', 'mt', 'ref', 'score', 'raw_score', 'annotators'])
		test_writer = csv.writer(test_file, delimiter=',')
		test_writer.writerow(['lp', 'src', 'mt', 'ref', 'score', 'raw_score', 'annotators'])

		# Extract all MT hypotheses that correspond to
		# 100 different source segments.
		dev_set = set()
		data_reader = csv.DictReader(infile, delimiter=',')
		for row in data_reader:
			src = row['src']
			if len(dev_set) < 100:
				dev_set.add(src)
			if src in dev_set:
				dev_writer.writerow(row.values())
			else:
				test_writer.writerow(row.values())
	print(len(dev_set))
	print(dev_set)


def main():
	with open('../prepared_data/2020-de-da.csv', 'r', encoding='utf-8') as infile:
		out_dev = '../prepared_data/2020-de-da-dev.csv'
		out_test = '../prepared_data/2020-de-da-test.csv'
		dev_test_split(infile, out_dev, out_test)


if __name__ == '__main__':
	main()