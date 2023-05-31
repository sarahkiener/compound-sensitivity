#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


import argparse
from pathlib import Path

"""
Compute the Average Score for Each MT System
============================================
After the segments of an MT system have been
scored, compute the system-level score for a 
given MT system by averaging over the segment
scores.

"""


def parse_args():
	parser = argparse.ArgumentParser(description="compute the average score per system")
	parser.add_argument('--metric', type=str, required=True, help="name of metric to evaluate")
	return parser.parse_args()


def compute_sys_scores():
	'''
	Retrieve the segment-level scores for each
	available MT system and average over them
	to obtain a system-level score.
	'''

	# Collect all segment-level scores for each system in a dictionary
	sys_dict = {}
	segfile_path = Path(f'results/{args.metric}.seg.score')
	with open(segfile_path, 'r', encoding='utf-8') as segfile:
		for line in segfile:
			metric, lp, testset, refset, system, docid, segid, score = line.strip().split('\t')
			if lp == 'en-de' and testset == 'newstest2020' and refset == 'newstest2020':
				if system in sys_dict:
					sys_dict[system].append(float(score))
				else: 
					sys_dict[system] = [float(score)]
					
	# Compute the average system-level score and write it to the outfile
	sysfile_path = Path(f'results/{args.metric}.sys.score')
	with open(sysfile_path, 'wt', encoding='utf-8') as outfile:
		sorted_sys = sorted(sys_dict.items())
		for sys, scores in sorted_sys:
			sys_score = sum(scores) / len(scores)
			sys_line = args.metric + '\t' + 'en-de' + '\t' + 'newstest2020' + '\t' + 'newstest2020' + '\t' + sys + '\t' + str(sys_score)
			outfile.write(sys_line + '\n')


def main(args):
	compute_sys_scores()


if __name__ == '__main__':
	args = parse_args()
	main(args)
