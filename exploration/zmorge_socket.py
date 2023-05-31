#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Master's Thesis Compound Sensitivity
# Author: Prof. Dr. Rico Sennrich 
# Modified by: Sarah Kiener


# Part of this script is copied from https://github.com/rsennrich/clevertagger/blob/4b35d11c7f9ef3ab50a6f5000175a89f59ea7aed/extract_features.py
# and adapted to the purposes of this thesis


"""
Morphological Analysis by Zmorge
================================
Send the unknown words to Zmorge for morphological
analysis. Classify the words into five categories 
according to their morphological properties.
"""


import sys
import os
import re
import socket
import time
import codecs
import argparse
from typing import List
from pathlib import Path
from subprocess import Popen, PIPE
from collections import defaultdict


# set variables
SFST_BIN = 'fst-infl2-daemon'
SMOR_MODEL = '../mbr/zmorge-20150315-smor_newlemma.ca'
SMOR_ENCODING = 'utf-8'
PORT = 9010


def parse_args():
	parser = argparse.ArgumentParser(description="Classify strange words into different categories")
	parser.add_argument('--metric', type=str, required=False, default='', help="name of metric or 'beam' for beam search outputs")
	return parser.parse_args()

class MorphAnalyzer():
	# class MorphAnalyzer is copied from https://github.com/rsennrich/clevertagger/blob/4b35d11c7f9ef3ab50a6f5000175a89f59ea7aed/extract_features.py
    """Base class for morphological analysis and feature extraction"""

    def __init__(self):
        
        self.posset = defaultdict(set)

        # Gertwol/SMOR only partially analyze punctuation. This adds missing analyses.
        for item in ['(',')','{','}','"',"'",u'”',u'“','[',']','«','»','-','‒','–','‘','’','/','...','--']:
            self.posset[item].add('$(')
        self.posset[','].add('$,')
        for item in ['.',':',';','!','?']:
            self.posset[item].add('$.')

        #regex to check if word is alphanumeric
        #we don't use str.isalnum() because we want to treat hyphenated words as alphanumeric
        self.alphnum = re.compile(r'^(?:\w|\d|-)+$', re.U)
        

class SMORAnalyzer(MorphAnalyzer):
	# class SMORAnalyzer is copied from https://github.com/rsennrich/clevertagger/blob/4b35d11c7f9ef3ab50a6f5000175a89f59ea7aed/extract_features.py#L182
    def __init__(self):
        MorphAnalyzer.__init__(self)

        #regex to get coarse POS tag from SMOR output
        self.re_mainclass = re.compile(u'<\+(.*?)>')
        self.PORT = PORT

        # start server, and make sure it accepts connection
        self.p_server = self.server()

    def server(self):
        """Start a socket server. If socket is busy, look for available socket"""

        while True:
            try:
                server = Popen([SFST_BIN, str(self.PORT), SMOR_MODEL], stderr=PIPE, bufsize=0)
            except OSError as e:
                if e.errno == 2:
                    sys.stderr.write('Error: {0} not found. Please install sfst and/or adjust SFST_BIN in clevertagger config.\n'.format(SFST_BIN))
                    sys.exit(1)
            error = b''
            while True:
                error += server.stderr.read(1)
                if error.endswith(b'listening to the socket ...'):
                    return server
                elif error.endswith(b'ERROR on binding'):
                    self.PORT += 1
                    sys.stderr.write('PORT {0} busy. Trying to use PORT {1}\n'.format(self.PORT-1, self.PORT))
                    break
                elif server.poll():
                    error += server.stderr.read()
                    error = error.decode('utf-8')
                    sys.stderr.write(error)
                    sys.exit(1)


    def client(self, words):
        """Communicate with socket server to obtain analysis of word list."""

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', self.PORT))
        s.sendall('\n'.join(words).encode(SMOR_ENCODING))
        s.shutdown(socket.SHUT_WR)
        analyses = b''
        data = True
        while data:
            data = s.recv(4096)
            analyses += data

        return analyses

    def convert(self, analyses):
        # written by Sarah Kiener
        """
        Receives the morphological analyses from Zmorge, extracts the lemma
        and classifies it into one of the five categories "non-word", 
        "non-noun/non-compound", "known compound", "hyphenated compound", "new compound"
        
        Args:
            analyses: 				List of morphological analyses provided by Zmorge

        Return:
        	non_words:				List of lemmata considered as non-words by Zmorge
        	no_compounds: 			List of lemmata that are neither nouns nor nominal compounds
        	hyphenated_compounds:	List of compounds with a hyphen
        	known_compounds:		List of lemmata considered as known compounds by Zmorge 
        	new_compounds:			List of novel compounds
        """

        result_dict = {}
        non_words = []
        no_compounds = []
        hyphenated_compounds = []
        known_compounds = []
        new_compounds = []

        # Decode the results from Zmorge and extract the lemma of each entry
        analyses = analyses.decode(SMOR_ENCODING).strip()[2:]
        entries = analyses.split('\n> ')
        for entry in entries:
            results = entry.split('\n')
            lemma = results[0]
            if lemma.startswith('>'):
                lemma = lemma[2:]
            # If Zmorge did not return a morphological analysis for a word,
            # add it to the category "non-words"
            if results[1].startswith('no result'):
                non_words.append(lemma)
            # If Zmorge returned at least one morphological analysis for a word,
            # add its lemma and all possible analyses to a dict
            else:
                result_dict[lemma] = results[1:]

        # Classify the lemmata into different categories by searching
        # for certain symbols in the Zmorge analyses
        morphs = ['<TRUNC>', '<#>', '<->', '<~>']
        for lemma, results in result_dict.items():
            known_compound = False
            no_noun = True
            no_compound = True
            hyphenated = False
            for result in results:
            	# Find nouns
                if '<+NN>' in result:
                    no_noun = False
                # Find compounds
                if '<#>' in result or '<TRUNC>' in result:
                    no_compound = False
                # Find hyphenated compounds
                if '<TRUNC>' in result:
                    hyphenated = True
                # Find known compounds: Zmorge returns at least one analysis
                # where the compounds forms an entity without any special symbol inserted
                if not any(morph in result for morph in morphs):
                    known_compound = True
            # Add the words into their respective category: "non-noun/non-compound",
            # "hyphenated compound", "known compound" or "new compound"
            if no_noun or no_compound:
                no_compounds.append(lemma)
            elif hyphenated:
            	hyphenated_compounds.append(lemma)
            elif known_compound:
            	known_compounds.append(lemma)
            else:
            	new_compounds.append(lemma)

        print('non-words:', len(non_words))
        print('no nouns or compounds:', len(no_compounds))
        print('hyphenated compounds:', len(hyphenated_compounds))
        print('known compounds:', len(known_compounds))
        print('new compounds:', len(new_compounds))

        return non_words, no_compounds, hyphenated_compounds, known_compounds, new_compounds




    def main(self, args):
        # Written by Sarah Kiener

        # Read in the unknown words from the outfile created in find_compounds.py
        # Add them to the ToDo-list.
        todo = []
        with open(Path(f'../compounds/unknown_words/unknown_{args.metric}.txt'), 'r', encoding='utf-8') as word_file:
            for line in word_file:
                line = line.strip()
                word, sent_num = line.split('\t')
                todo.append(word)
        # Deduplicate the word list
        todo = list(set(todo))
        print('words to analyse:', len(todo))
        
        # Send the word list to Zmorge for morphological analysis.
        # Classify the results into different categories.
        try:
            analyses = self.client(todo)
            non_words, no_compounds, hyphenated_compounds, known_compounds, new_compounds = self.convert(analyses)
            
            # Write the different categories of words into separate outfiles
            with open(Path(f'../compounds/unknown_words_per_category/non_words/non_words_{args.metric}.txt'), 'w', encoding='utf-8') as non_words_file:
                for word in non_words:
                    non_words_file.write(word + '\n')

            with open(Path(f'../compounds/unknown_words_per_category/non_nouns/non_nouns_{args.metric}.txt'), 'w', encoding='utf-8') as no_compounds_file:
                for word in no_compounds:
                    no_compounds_file.write(word + '\n')

            with open(Path(f'../compounds/unknown_words_per_category/hyphenated_compounds/hyphenated_compounds_{args.metric}.txt'), 'w', encoding='utf-8') as hyphenated_compounds_file:
                for word in hyphenated_compounds:
            	    hyphenated_compounds_file.write(word + '\n')

            with open(Path(f'../compounds/unknown_words_per_category/known_compounds/known_compounds_{args.metric}.txt'), 'w', encoding='utf-8') as known_compounds_file:
                for word in known_compounds:
                    known_compounds_file.write(word + '\n')

            with open(Path(f'../compounds/unknown_words_per_category/new_compounds/new_compounds_{args.metric}.txt'), 'w', encoding='utf-8') as new_compounds_file:
                for word in new_compounds:
                    new_compounds_file.write(word + '\n')
		    
        finally:
            self.p_server.terminate()
        


if __name__ == '__main__':
    args = parse_args()
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

    Analyzer = SMORAnalyzer()
    Analyzer.main(args)
    
