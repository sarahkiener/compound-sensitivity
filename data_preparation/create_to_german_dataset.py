#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Extract German Segments from WMT Data Sets
==========================================
From the WMT Metrics Shared Task data sets, 
extract the German segments. Save the segments
of the years 2017 - 2019 in one csv-file as 
training set. Save the segments of 2020 in 
another csv-file that is later divided into
validation and test set.

"""


import glob
import csv
import os
from pathlib import Path



def create_to_german_dataset(directory: str):
	"""
	Search the directory for WMT data sets.
	Extract the German segments and write them 
	to the output csv-file.

	Args:
		directory:	Path to the directory with the WMT Metrics Shared Task data sets

	"""
	
	# Open and prepare the output csv. 
	# To extract the validation segments, change the filename to '/2020-de-da.csv'.
	with open(directory + '2017-18-19-de-da.csv', 'wt', encoding='utf-8') as de_file:
		de_writer = csv.writer(de_file, delimiter=',')
		de_writer.writerow(['lp', 'src', 'mt', 'ref', 'score', 'raw_score', 'annotators'])	

		counter = 0
		counts_per_year = []
		# Search the directory for WMT csv-files and extract German segments
		os.chdir(directory)
		# For the validation segments, change the file name to '2020-da.csv'
		for file in glob.glob('201?-da.csv'):
			with open(file, 'r', encoding='utf-8') as infile:
				print(file)
				data_reader = csv.DictReader(infile, delimiter=',')
				for row in data_reader:
					lp = row['lp']
					if '-de' in lp:
						de_writer.writerow(row.values())
						counter += 1

			counts_per_year.append(counter)
			print('sentences:', counter)
			counter = 0
	print('total sentences:', sum(counts_per_year))



def main():
	path = '../prepared_data'
	create_to_german_dataset(path)


if __name__ == '__main__':
	main()