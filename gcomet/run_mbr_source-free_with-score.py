#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener

"""
Run MBR Decoding with a Source-Free or Reference-Free 
COMET Metric and Write Scores to JSON File
======================================================
The MBR implementation by Chantal Amrhein is modified 
to enable decoding with only two input segments: an 
MT hypothesis and either a source or a reference. 
In addition, the script is modified in order that each 
candidate and its average score are saved in a json file. 
This method is used to combine two metrics in MBR decoding.

"""


import json
import argparse
import itertools

import numpy as np
from collections import defaultdict
from comet import load_from_checkpoint

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--transl_file', type=argparse.FileType('r'), required=True, help='path to file containing sampled support hypotheses (number of lines should be a multiple of the number of lines in src file).')
    ap.add_argument('-c', '--candidate_file', type=argparse.FileType('r'), required=True, help='path to file containing sampled candidate sentences (number of lines should be a multiple of the number of lines in src file).')
    ap.add_argument('-j', '--json_file', type=argparse.FileType('w'), required=True, help='path to output json file with the average score per candidate.')
    ap.add_argument('-ns', '--n_support', type=int, required=True, default=1, help='number of support samples per src sentence.')
    ap.add_argument('-nc', '--n_candidates', type=int, required=True, default=1, help='number of candidate samples per src sentence.')
    ap.add_argument('-b', '--batch_size', type=int, required=False, default=8, help='batch size for MBR decoding.')
    ap.add_argument('-g', '--gpus', type=int, required=False, default=1, help='how many GPUs to use, 0 == CPU.')
    ap.add_argument('-m', '--model_name', type=str, required=False, default='models/gcomet-wwm/epoch=2-step=11130.ckpt', help='COMET model name.')
    return ap.parse_args()

def chunk(iterator, size):
    while True:
        chunk = list(itertools.islice(iterator, size))
        if chunk:
            yield chunk
        else:
            break

def main(args):
    # Load model from checkpoint
    model_path = args.model_name
    print(model_path)
    model = load_from_checkpoint(model_path)

    data = []
    score_dict = defaultdict(dict)
    idx = 0

    # Prepare the data    
    for candidates, support in zip(chunk(args.candidate_file, args.n_candidates), chunk(args.transl_file, args.n_support)):
        example = {}
        candidates = [c.strip() for c in candidates]
        support = [s.strip() for s in support]
        example['mt'] = candidates
        example['src'] = support
        data.append(example)

    # Compute the utility scores
    batched_matrices = model.get_utility_scores(data, args.batch_size, gpus=args.gpus)

    # Calculate the average score per candidate and write the candidate and its score
    # to the json output file
    for matrices, examples in zip(batched_matrices, chunk((x for x in data), args.batch_size)):
        if args.gpus > 0:
            matrices = matrices.cpu()
        matrices = np.reshape(matrices, [len(examples), len(examples[0]['mt']), -1])

        for matrix, example in zip(matrices, examples):
            mbr_scores = np.average(matrix, axis=-1)
            for score, candidate in (zip(mbr_scores, example['mt'])):
                score_dict[idx][candidate] = float(score)
            idx += 1

    json.dump(score_dict, args.json_file, indent=4, sort_keys=True)


if __name__ == '__main__':

    args = parse_args()
    main(args)
