#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Authors: Chantal Amrhein, Sarah Kiener


"""
MBR-Based Sensitivity Analysis with a Soure-Free or Reference-Free COMET Metric
===============================================================================
The MBR-based sensitivity analysis by Chantal Amrhein is modified to enable 
decoding with only two input segments: an MT hypothesis and
either a source or a reference.

"""


import argparse
import itertools
import json

import numpy as np
from comet import load_from_checkpoint

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--ref_file', type=argparse.FileType('r'), required=True, help='path to file containing the support sentences (number of lines may be a multiple of the number of src sentences).')
    ap.add_argument('-j', '--json_file', type=argparse.FileType('r'), required=True, help='path to json file containing source sentences and variable numbers of candidate sentences.')
    ap.add_argument('-o', '--output_file', type=argparse.FileType('w'), required=True, help='path to output file.')
    ap.add_argument('-ns', '--n_support', type=int, required=True, default=1, help='number of support samples per src sentence.')
    ap.add_argument('-b', '--batch_size', type=int, required=False, default=8, help='batch size for MBR decoding.')
    ap.add_argument('-g', '--gpus', type=int, required=False, default=1, help='how many GPUs to use, 0 == CPU.')
    ap.add_argument('-m', '--model_name', type=str, required=False, default='models/gcomet-wwm/epoch=2-step=11130.ckpt', help='path to COMET model')
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
    sentences = json.load(args.json_file)

    # Prepare the data
    for (i, sentence), support in zip(sentences.items(), chunk(args.ref_file, args.n_support)):
        example = {}
        example['id'] = i
        support = [s.strip() for s in support]
        example['mt'] = sentence.values()
        example['src'] = support
        data.append(example)

    # Compute the utility scores
    batched_matrices = model.get_utility_scores(data, args.batch_size, gpus=args.gpus)
 
    # Write the candidates and their sensitivity scores to a json file
    results = {}
    for matrices, examples in zip(batched_matrices, chunk((x for x in data), args.batch_size)):
        if args.gpus > 0:
            matrices = matrices.cpu()
        matrices = np.reshape(matrices, [len(examples), len(examples[0]['mt']), -1])

        for matrix, example in zip(matrices, examples):
            mbr_scores = np.average(matrix, axis=-1)
            result = {}
            for k, score in zip(sentences[example['id']].keys(), mbr_scores):
                result[k] = (sentences[example['id']][k], str(score))
            results[example['id']] = result

    json.dump(results, args.output_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    args = parse_args()
    main(args)
