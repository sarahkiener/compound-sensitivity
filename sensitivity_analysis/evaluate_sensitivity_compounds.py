#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Calculate Sensitivity Scores for Compounds
===========================================
To obtain the sensitivity scores, calculate
the difference between the score assigned 
to the correct reference and the perturbed
candidate. The differences are then averaged 
per error type.

"""


import argparse
import json
from collections import defaultdict

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', type=str, required=True, help='path to json file with scores per candidate')
    ap.add_argument('-o', '--order', type=str, nargs='+', required=False, help='desired output order of error types')
    return ap.parse_args()


def main(args):

    with open(args.file, 'r', encoding='utf-8') as infile:
        results = json.load(infile)

    scores = defaultdict(list)

    for _, result in results.items():
        error_types = result.keys()
        refscore = float(result['ref'][1])

        # Calculate the score difference between the reference and the perturbed candidate
        for error_type in error_types:
            if error_type == 'ref':
                continue
            elif "most_freq" in error_type:
                scores["most_freq"].append(refscore - float(result[error_type][1]))
                scores["compounds"].append(refscore - float(result[error_type][1]))
            elif "most_sim" in error_type:
                scores["most_sim"].append(refscore - float(result[error_type][1]))
                scores["compounds"].append(refscore - float(result[error_type][1]))
            else:
                scores[error_type].append(refscore - float(result[error_type][1]))


    order = sorted(scores) if not args.order else args.order

    # Average the score differences per error type
    for error_type in order:
        print(f'{error_type}\t{str(np.mean(scores[error_type]))}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
