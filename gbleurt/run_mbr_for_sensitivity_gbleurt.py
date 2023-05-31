#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
MBR-Based Sensitivity Analysis with GBLEURT
===========================================
Run an MBR-based sensitivity analysis with
GBLEURT given an MT hypothesis and a reference.

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
    parser = argparse.ArgumentParser(description="MBR-decoding with BLEURT metric on GBERT model")
    parser.add_argument('-m', '--model_name', type=str, required=True, help='trained GBLEURT model name.')
    parser.add_argument('-r', '--ref_file', type=argparse.FileType('r'), required=True, help='path to file containing the support sentences (number of lines may be a multiple of the number of src sentences).')
    parser.add_argument('-j', '--json_file', type=argparse.FileType('r'), required=True, help='path to json file containing source sentences and variable numbers of candidate sentences.')
    parser.add_argument('-o', '--output_file', type=argparse.FileType('w'), required=True, help='path to output file.')
    parser.add_argument('-ns', '--n_support', type=int, required=True, help='number of support samples per src sentence.')
    return parser.parse_args()


def chunk(iterator, size):
    '''
    Iterate through the dataset.

    Args:
        iterator:   File with the data samples
        size:       Number of candidate / support sentences per source sentence
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
        data_dict:  Dictionary with sorted candidate and support lists

    Returns:
        dataframe: pandas.DataFrame with candidate and support sentences
    '''
    mt_sents = []
    ref_sents = []   
    for candidate in data_dict['mt']:
        for support in data_dict['ref']:
            mt_sents.append(candidate)
            ref_sents.append(support)
    mbr_data = {'mt' : mt_sents, 'ref' : ref_sents}
    dataframe = pd.DataFrame(mbr_data)
    return dataframe
    


def prepare_data(dataframe, tokenizer):
    '''
    Prepare and encode the dataset. 
    
    Args:
        dataframe:          Dataframe with input data
        tokenizer:          Huggingface tokenizer

    Return:
        encoded_dataset:    Dataset with encoded data
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
    sentences = json.load(args.json_file)
    
    # Create a list of dictionaries with one dictionary per source sentence
    # Each dictionary is of the form: 
    # {'id' : '1', mt' : dict_values([candidate_1, candidate_2, ..., candidate_n]), 'ref' : [support_1, support_2, ..., support_n]}
    for (i, sentence), support in zip(sentences.items(), chunk(args.transl_file, args.n_support)):
        example = {}
        example['id'] = i
        support = [s.strip() for s in support]
        example['mt'] = sentence.values()
        example['ref'] = support
        data.append(example)
    
    results ={}
    
    # For each source sentence, build a dataframe containing all possible candidate-support combinations
    for data_dict in data:  
        dataframe = build_dataframe(data_dict)    
        # Score each sentence pair
        preds = predict(dataframe)        
        # Reshape the prediction array to have n_candidate rows and n_support columns
        preds_per_candidate = preds.reshape((len(data_dict['mt']), args.n_support))               
        # Calculate the average score across each row, i.e. per candidate
        mbr_scores = np.average(preds_per_candidate, axis=1)
        # Save each candidate with its average score in the results dictionary
        result = {}
        for k, score in zip(sentences[data_dict['id']].keys(), mbr_scores):
            result[k] = (sentences[data_dict['id']][k], str(score))
        results[data_dict['id']] = result

    json.dump(results, args.output_file, ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)