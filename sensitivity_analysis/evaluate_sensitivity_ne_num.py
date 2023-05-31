#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Chantal Amrhein, Sarah Kiener


"""
Calculate Sensitivity Scores for Nouns, NEs and Numbers
=======================================================
To obtain the sensitivity scores, calculate the difference 
between the score assigned to the correct reference 
and the perturbed candidate. The differences are then 
averaged per error type.
The script by Chantal Amrhein is modified to not consider
the absolute difference, but rather the non-absolute
difference.

"""


import argparse
import json
from collections import defaultdict

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', type=str, required=True,
                    help='path to results json file')
    ap.add_argument('-o', '--order', type=str, nargs='+', required=False,
                    help='desired output order of error types')
    return ap.parse_args()


def main(args):

    with open(args.file, 'r') as infile:
        results = json.load(infile)

    scores = defaultdict(list)

    for _, result in results.items():
        error_types = result.keys()
        refscore = float(result['reference'][1])
       
        for error_type in error_types:
            if error_type == 'reference' or result[error_type][0] == '':
                continue
            # Calculate the score difference between the reference and the perturbed candidate
            scores[error_type].append(refscore - float(result[error_type][1]))

    order = sorted(scores) if not args.order else args.order

    # Average the score differences per error type
    for type in order:
        print(f'{type}\t{str(np.mean(scores[type]))}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
