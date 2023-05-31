#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener

"""
Combine Two Metrics in MBR Decoding
=======================================
Read in the json files with the scores
assigned to each candidate by two 
different metrics. Calculate the average
score and write the candidate with the
highest average score to the output file.

"""


import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MBR-Decoding: find best candidate by combining two metrics")
    parser.add_argument('-o', '--output_file', type=argparse.FileType('w'), required=True, help='path to output file.')
    parser.add_argument('-j1', '--json_file_1', type=argparse.FileType('r'), required=True, help='path to json file with the scores per candidate assigned by metric 1')
    parser.add_argument('-j2', '--json_file_2', type=argparse.FileType('r'), required=True, help='path to json file with the scores per candidate assigned by metric 2')
    return parser.parse_args()

def combine_metrics():
    # Read in the candidates and their scores assigned by each metric
    score_dict_1 = json.load(args.json_file_1)
    score_dict_2 = json.load(args.json_file_2)
    
    # Calculate the average score per candidate and write the 
    # candidate with the highest average score to the output file
    for i in range(len(score_dict_1)):
        combi_dict = {}
        for candidate, score in score_dict_1[str(i)].items():
    	    combi_dict[candidate] = (score + score_dict_2[str(i)][candidate]) / 2
        best = max(combi_dict, key=combi_dict.get)
        args.output_file.write(best + '\n')


def main(args):
    combine_metrics()


if __name__ == '__main__':
    args = parse_args()
    main(args)