#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
MBR Decoding with GBLEURT
=========================
Run MBR decoding with GBLEURT. 
The script supports both:
	1) 	Directly find the best candidate and write it to the outfile
	2) 	Write the scores for each candidate-support pair to a json file. 
		Run mbr_find_best_candidate.py to later select the best candidate.

"""



import json
import argparse
import itertools
import pandas as pd
import numpy as np 
from datasets import Dataset
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="MBR decoding with GBLEURT")
    parser.add_argument('-m', '--model_name', type=str, required=True, help='trained GBLEURT model name.')
    parser.add_argument('-t', '--transl_file', type=argparse.FileType('r'), required=True, help='path to file containing sampled support hypotheses (number of lines should be a multiple of the number of lines in src file).')
    parser.add_argument('-c', '--candidate_file', type=argparse.FileType('r'), required=True, help='path to file containing sampled candidate sentences (number of lines should be a multiple of the number of lines in src file).')
    parser.add_argument('-o', '--output_file', type=argparse.FileType('w'), required=True, help='path to output file.')
    parser.add_argument('-j', '--json_file', type=argparse.FileType('w', encoding='utf-8'), required=True, help='path to output json file to store the scores per candidate-support pair.')
    parser.add_argument('-ns', '--n_support', type=int, required=True, help='number of support samples per src sentence.')
    parser.add_argument('-nc', '--n_candidates', type=int, required=True, help='number of candidate samples per src sentence.')
    parser.add_argument('-ic', '--include_candidate', type=bool, required=False, action='store_true', help='include the comparison of the candidate to itself in MBR decoding')
    return parser.parse_args()


def chunk(iterator, size):
    '''
    Iterate through the dataset.

    Args:
        iterator:	File with the data samples
        size:		Number of candidate / support sentences per source sentence
    '''
    while True:
        chunk = list(itertools.islice(iterator, size))
        if chunk:
            yield chunk
        else:
            break


def build_dataframe(data_dict):
    '''
    Create a dataframe with all possible candidate-support pairs for a given source sentence.

    Args:
        data_dict:	Dictionary with sorted candidate and support lists

    Returns:
        dataframe: pandas.DataFrame with candidate and support sentences
    '''
    mt_sents = []
    ref_sents = []   
    for candidate in data_dict['mt']:
        for support in data_dict['ref']:
        	if not args.include_candidate and candidate == support:
                continue                   
            mt_sents.append(candidate)
            ref_sents.append(support)
    mbr_data = {'mt' : mt_sents, 'ref' : ref_sents}
    dataframe = pd.DataFrame(mbr_data)
    return dataframe


def prepare_data(dataframe, tokenizer):
    '''
    Prepare and encode the dataset. 
	
    Args:
        dataframe:			Dataframe with input data
        tokenizer:			Huggingface tokenizer

    Return:
        encoded_dataset:	Dataset with encoded data
    '''

    dataset = Dataset.from_pandas(dataframe, preserve_index=False)
    # Tokenize and encode the data
    encoded_dataset = dataset.map(lambda example : tokenizer(example['ref'], example['mt'], padding='max_length', max_length=512, truncation=True), batched=True)
    return encoded_dataset


def predict(dataframe):
    '''
    Load the trained GBLEURT model to predict quality scores 
    for German MT outputs compared to a reference translation.

    Args: 
        dataframe:    Dataframe with input data

    Return:
        raw_pred:     Predictions of the model
    '''
    # Load the trained model and tokenizer
    model_dir = 'models/' + args.model_name
    trained_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    trained_tokenizer = AutoTokenizer.from_pretrained(model_dir)

    trainer = Trainer(model=trained_model)

    # Encode the dataset
    encoded_dataset = prepare_data(dataframe, trained_tokenizer)

    # Predict the score for each sentence pair
    raw_pred, label_id, metric = trainer.predict(encoded_dataset)
    return raw_pred  


def main(args):	
    data = []
    # Create a list of dictionaries with one dictionary per source sentence.
    # Each dictionary is of the form: 
    # {'mt' : [candidate_1, candidate_2, ..., candidate_n], 'ref' : [support_1, support_2, ..., support_n]}
    for candidates, support in zip(chunk(args.candidate_file, args.n_candidates), chunk(args.transl_file, args.n_support)):
        example = {}
        candidates = [c.strip() for c in candidates]
        support = [s.strip() for s in support]
        example['mt'] = candidates
        example['ref'] = support
        data.append(example)
    
    score_dict = defaultdict(list)
    src_count = 0
    
    # For each source sentence, build a dataframe containing all possible candidate-support combinations
    for data_dict in data:
        dataframe = build_dataframe(data_dict)
        # Score each sentence pair
        preds = predict(dataframe)      
        # Reshape the prediction array to have n_candidate rows and n_support (-1) columns
        preds_per_candidate = preds.reshape((args.n_candidates, args.n_support)) # add -1 if candidate is not compared to itself     
 
        # Directly find the best candidate:        
        # Calculate the average score across each row, i.e. per candidate
        mbr_scores = np.average(preds_per_candidate, axis=1)
        # Find the candidate with the maximum score
        prediction_idx = np.argmax(mbr_scores)
        # Retrieve the best candidate from the data_dict and write it to the outfile
        args.output_file.write(data_dict['mt'][prediction_idx] + '\n')
       
    
        # Store the score for each candidate-support pair in a json file of the form:
        # {sent_id : [{cand_1 : [{support_1 : score_1}, {support_2 : score_2}, ..., {support_n : score_n}]}, 
        # 			 {cand_2 : [{support_1 : score_1}, {support_2 : score_2}, ..., {support_n : score_n}]}, ...]}
        for i in range(args.n_candidates):
        	cand_dict = defaultdict(list)
        	for j in range(args.n_support):
        		cand_dict[data_dict['mt'][i]].append({data_dict['ref'][j] : float(preds_per_candidate[i][j])})
        	score_dict[src_count].append(dict(cand_dict))
        src_count += 1
    score_dict = dict(score_dict)
    json.dump(score_dict, args.json_file, indent=4, sort_keys=True)
    


if __name__ == '__main__':
    args = parse_args()
    main(args)