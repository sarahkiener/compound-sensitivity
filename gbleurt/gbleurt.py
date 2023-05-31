#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Sarah Kiener


"""
Train a BLEURT model on top of GBERT
====================================
Train a regression model corresponding to
the BLEURT metric on top of GBERT embeddings.
Use the metric for quality score prediction.
Two variants of the German BLEURT models are trained:
1) with GBERT-WWM embeddings
2) with GBERT-SWM embeddings

"""


import argparse
import pandas as pd
import numpy as np 
from random import shuffle
from datasets import Dataset, DatasetDict, load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer



def parse_args():
	parser = argparse.ArgumentParser(description="Train a BLEURT metric on GBERT model")
	parser.add_argument('--train', action='store_true', required=False, help="train a new model")
	parser.add_argument('--predict', action='store_true', required=False, help="load existing model for predictions")
	parser.add_argument('--gbert_checkpoint', type=str, required=False, default='deepset/gbert-base', help="GBERT model for embeddings")
	parser.add_argument('--model_name', type=str, required=True, help='GBLEURT model name.')
	parser.add_argument('--train_data', type=str, required=False, default='../prepared_data/2017-18-19-de-da.csv', help="path to training data")
	parser.add_argument('--dev_data', type=str, required=False, default='../prepared_data/2020-de-da-dev.csv', help="path to dev data")
	parser.add_argument('--test_data', type= str, required=False, default='../prepared_data/2020-de-da-official-testset.csv', help="path to test data")
	return parser.parse_args()



def train():
	"""
	Train a model according to the BLEURT metric
	on top of GBERT embeddings (WWM or SWM).
	Use Direct Assessment data for training.
	The final model scores the quality of German MT 
	outputs compared to a reference translation.
	"""

	model_dir = 'models/' + args.model_name
	# Tokenize and encode the train and dev data
	tokenizer = AutoTokenizer.from_pretrained(args.gbert_checkpoint)
	train_dataset = prepare_data(args.train_data, tokenizer)
	dev_dataset = prepare_data(args.dev_data, tokenizer)
	encoded_dataset = DatasetDict()
	encoded_dataset['train'] = train_dataset
	encoded_dataset['test'] = dev_dataset
	print(encoded_dataset)
	
	# Load model and add regression layer
	# num_labels=1 and problem_type="regression" for regression problems
	model = AutoModelForSequenceClassification.from_pretrained(args.gbert_checkpoint, num_labels=1, problem_type="regression")

	# Train the model
	training_args = TrainingArguments(
                                  output_dir=model_dir,
                                  optim="adamw_hf",
                                  learning_rate=1e-5,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  gradient_accumulation_steps=4,
                                  eval_steps=500, # original BLEURT uses 1500 eval_steps
                                  evaluation_strategy="steps",
                                  logging_strategy="steps",
                                  logging_steps=500,
                                  save_total_limit=2,
                                  save_steps=500,
                                  max_steps=20000, # original BLEURT uses 40000 steps for fine-tuning
                                  load_best_model_at_end=True,
                                  ) 

	trainer = Trainer(
                  model=model,
                  args=training_args,
                  train_dataset=encoded_dataset["train"],
                  eval_dataset=encoded_dataset["test"],
                  compute_metrics=compute_metrics
                  )

	trainer.train()

	# Save the model and tokenizer
	model.save_pretrained(model_dir)
	tokenizer.save_pretrained(model_dir)


def predict():
	"""
	Load the trained GBLEURT model to predict
	quality scores for German MT outputs
	compared to a reference translation.
	"""

	# Load the trained model
	model_dir = 'models/' + args.model_name
	trained_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	trained_tokenizer = AutoTokenizer.from_pretrained(model_dir)
	trainer = Trainer(model=trained_model)

	# Tokenize and encode the test data
	encoded_testset = prepare_data(args.test_data, trained_tokenizer, testing=True)
	print(encoded_testset)

	# Predict
	raw_pred, label_id, metric = trainer.predict(encoded_testset)
	# Save predictions to output files
	save_results(raw_pred)




def prepare_data(datafile, tokenizer, testing=False):
	"""
	Prepare and encode the dataset. 
	
	Args:
		datafile:			File with input data
		tokenizer:			Huggingface tokenizer
		testing: 			If False: 	Prepare data for training: Discard segments > 512 (max_length of GBERT)
							If True: 	Prepare data for testing: Truncate segments > 512 tokens (max_length of GBERT)

	Return:
		encoded_dataset:	Dataset with encoded data

	"""

	de_da_data = pd.read_csv(datafile, sep=',')
	# Truncate segments > 512 tokens for testing
	if testing:
		de_prepared_data = de_da_data.drop(['lp', 'src'], axis=1)
		dataset = Dataset.from_pandas(de_prepared_data, preserve_index=False) 
		# Tokenize and encode the data
		encoded_dataset = dataset.map(lambda example : tokenizer(example['ref'], example['mt'], padding='max_length', max_length=512, truncation=True), batched=True)
	
	# Remove segments > 512 tokens for training
	else:
		de_prepared_data = de_da_data.drop(['lp', 'src', 'raw_score', 'annotators'], axis=1)
		de_prepared_data.rename(columns={'score': 'label'}, inplace=True)
		dataset = Dataset.from_pandas(de_prepared_data, preserve_index=False)
		# Tokenize and encode the data
		encoded_dataset = dataset.map(lambda example : tokenizer(example['ref'], example['mt'], padding='max_length', max_length=512, truncation=False), batched=True)
		# Filter out sentences with > 512 tokens
		encoded_dataset = encoded_dataset.filter(lambda example: len(example['input_ids']) <= 512)
		# Shuffle the data split
		encoded_dataset = encoded_dataset.shuffle()
	return encoded_dataset


def compute_metrics(eval_pred):
	'''
	Calculate the Mean Squared Error on the predictions and gold standard scores.

	Args:
		eval_pred:	Tuple with predections and gold standard labels

	Return:
		mse: 		Mean Squared Error
	'''
	
	predictions, labels = eval_pred
	mse_metric = load_metric("mse")
	mse = mse_metric.compute(predictions=predictions, references=labels)
	return mse


def save_results(raw_pred):
	'''
	Write the predictions into a output file.
	The file uses the format required by the official WMT 2020 Metrics Shared Task.

	Args: 
		raw_pred:	List with predictions by the model
	'''

	combi_path = 'combis/' + args.model_name + '_complete_mt_ref_combis.tsv'
	result_path = 'results/' + args.model_name + '.seg.score'
	with open(combi_path, 'r', encoding='utf-8') as combi_file, open(result_path, 'w', encoding='utf-8') as result_file:
		for line, score in zip(combi_file, raw_pred):
			line = line.strip()
			result_line = line + '\t' + str(score[0]) + '\n'
			result_file.write(result_line)



def main(args):
	if args.train:
		train()

	if args.predict:
		predict()



if __name__ == '__main__':
	args = parse_args()
	main(args)
