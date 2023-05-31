#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Predict Scores for Test Set with COMET
======================================
Predict the segment-level scores for 
the test set with a COMET-based model.
Save the scores to an output file.

"""


import argparse
import pandas as pd
from comet import download_model, load_from_checkpoint


def parse_args():
	parser = argparse.ArgumentParser(description="Scoring the test set with GCOMET")
	parser.add_argument('--model_name', type=str, required=True, help='GCOMET model name.')
	parser.add_argument('--model_path', type=str, required=True, help='Path to GCOMET model')
	parser.add_argument('--test_data', type= str, required=False, default='data/2020-de-da-official-testset.csv', help="path to test data")
	return parser.parse_args()


def predict_scores(model_path):
	'''
	Read in the test set and predict the segment-level and system-level scores

	Args:
		model_path:	Path to model checkpoint
	'''
	model = load_from_checkpoint(model_path)

	data = []

	# Read in the test set
	with open(args.test_data, 'r', encoding='utf-8') as test_file:
		test_data = pd.read_csv(test_file, sep=',')
		src = test_data["src"].astype(str)
		mt = test_data["mt"].astype(str)

		for src_sent, mt_sent in zip(src, mt):
			elem = {"src" : src_sent, "mt": mt_sent}
			data.append(elem)

		# Predict the segment-level and system-level scores
		seg_scores, sys_score = model.predict(data, batch_size=8, gpus=1)
		return seg_scores, sys_score


def save_results(seg_scores):
	'''
	Save the results in the format required by the WMT 2020 Metrics Shared Task

	Args:
		seg_scores:		segment-level scores predicted by COMET
	'''
	combi_path = '../wmt20_metrics_evaluation/combis/' + args.model_name + '_complete_mt_ref_combis.tsv'
	result_path = '../wmt20_metrics_evaluation/results/' + args.model_name + '.seg.score'
	with open(combi_path, 'r', encoding='utf-8') as combi_file, open(result_path, 'w', encoding='utf-8') as result_file:
		for line, score in zip(combi_file, seg_scores):
			line = line.strip()
			result_line = line + '\t' + str(score) + '\n'
			result_file.write(result_line)


def main(args):
	model_path = args.model_path
	seg_scores, sys_score = predict_scores(model_path)
	save_results(seg_scores)
	


if __name__ == '__main__':
	args = parse_args()
	main(args)