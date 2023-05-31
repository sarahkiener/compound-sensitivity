#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener

"""
MBR Decoding from JSON File with Scores
=======================================
Find the best candidate among the scored hypothesis.
This scripts supports two methods to run MBR decoding:
	1)	Include the comparison of a candidate to itself
		when calculating the average score per candidate.
	2)	Exclude the comparison of a candidate to itself
		when calculating the average score per candidate. 

"""


import json
import argparse
import numpy as np 


def parse_args():
    parser = argparse.ArgumentParser(description="MBR-Decoding: find best candidate among scored candidates")
    parser.add_argument('-o', '--output_file', type=argparse.FileType('w'), required=True, help='path to output file.')
    parser.add_argument('-j', '--json_file', type=argparse.FileType('r'), required=True, help='path to output json file to store the scores per candidate-support pair.')
    parser.add_argument('-ic', '--include_candidate', action='store_true', required=False, help="include the candidate itself in the MBR comparison")
    return parser.parse_args()



def main(args):
    # Load the data from the json file
    score_dict = json.load(args.json_file)

    if args.include_candidate:
        # Store the scores in a matrix per source sentence: each row corresponds to a candidate
        for idx in sorted(score_dict, key=int):
            preds_per_candidate = [[score for supp_dict in supp_list for score in supp_dict.values()] for cand_dict in score_dict[idx] for supp_list in cand_dict.values()]
            # Convert the list to an numpy array
            preds_per_candidate = np.array(preds_per_candidate)
            # Calculate the average score across each row, i.e. per candidate
            mbr_scores = np.average(preds_per_candidate, axis=1)
            # Find the candidate with the maximum score
            prediction_idx = np.argmax(mbr_scores)
            # Retrieve the best candidate from the score_dict and write it to the outfile
            for cand in score_dict[idx][prediction_idx]:
                args.output_file.write(cand + '\n')

    else:
        # Store the scores in a matrix per source sentence: each row corresponds to a candidate
        for idx in sorted(score_dict, key=int):
            preds_per_candidate = [[score for supp_dict in supp_list for score in supp_dict.values()] for cand_dict in score_dict[idx] for supp_list in cand_dict.values()] 
            # Convert the list to an numpy array
            preds_per_candidate = np.array(preds_per_candidate)
            # Exclude the score from comparing the candidate to itself (i.e. the diagonal of the matrix)
            preds_excl_cand = []
            for i in range(len(preds_per_candidate)):
                preds_excl_cand_row = [preds_per_candidate[i][j] for j in range(len(preds_per_candidate[i])) if i != j]
                preds_excl_cand.append(preds_excl_cand_row)
            preds_excl_cand = np.array(preds_excl_cand)
            # Calculate the average score across each row, i.e. per candidate
            mbr_scores = np.average(preds_excl_cand, axis=1)
            # Find the candidate with the maximum score
            prediction_idx = np.argmax(mbr_scores)
            # Retrieve the best candidate from the score_dict and write it to the outfile
            for cand in score_dict[idx][prediction_idx]:
                args.output_file.write(cand + '\n')
      


if __name__ == '__main__':
	args = parse_args()
	main(args)
