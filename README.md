# Compound Sensitivity of Neural Metrics
This repository contains the scripts and materials that were created as part of the Master's thesis *Schneeschrauben, Himmelbäume, Gartengebäcke: Investigating and Enhancing the Sensitivity of Trained Neural Metrics to German Compounds*.
The thesis investigates how sensitive trained neural metrics are towards German compounds. To shed light on the sensitivity of a metric and to uncover its blind spots, Minimum Bayes Risk (MBR) decoding is used as proposed by [Amrhein and Sennrich (2022)](https://aclanthology.org/2022.aacl-main.83/).

The first part of the thesis focuses on a case study on COMET-20. A semi-automatic analysis of the translations produced via MBR decoding with COMET-20 as utility function reveals that COMET-20 is not sensitive enough to compounds. It generates more nonsensical compounds than surface overlap-based utility functions. The exploration of the MBR-decoded outputs and the results of the analysis are described in the section [Exploration](#Exploration).

The second part of the thesis is devoted to strategies that may enhance the sensitivity of a neural metric towards German compounds. Mainly, the thesis investigates the effect of WWM during pre-training of the underlying language models on the sensitivity of a neural metric towards German compounds and other linguistic phenomena. To this end, [new metric models](#Training-of-New-Metrics) have been trained.

The newly trained metrics [GBLEURT](#GBLEURT) and [GCOMET](#GCOMET) are described below. They can be downloaded from Google Drive. Unpack the `.tar` archive and place the two GBLEURT models in `gbleurt/models` and the three COMET-based models in `gcomet/models`. The section [MBR-Based Sensitivity Analysis](#MBR-Based-Sensitivity-Analysis) outlines the method used to measure the sensitivity of the studied metrics towards certain linguistic phenomena, i.e. compounds, nouns, named entities and numbers.



## Data Sets
The data sets used in this thesis are saved in the folder `prepared_data`. 

All the experiments with MBR decoding are based on the `en-de` portion of the test set from the [WMT 2021 News Shared Task](https://aclanthology.org/2021.wmt-1.73/). The English source segments are contained in `prepared_data/en-de.src`, the German references (two references per source segment) can be found in `prepared_data/en-de.refs`. 

Sampling-based MBR decoding requires a candidate pool and a support set. In my thesis, I used the same candidates as Amrhein and Sennrich (2022) that they generated with their translation model. Throughout the thesis, the candidate and support sets were identical, containing 100 samples per source segment. These samples can be found in `prepared_data/en-de.samples`.

For the comparison with beam search translations, Amrhein and Sennrich (2022) provided me with their 1-best beam search outputs that are saved in `prepared_data/en-de.beam1`.

The new metrics were trained on the to-German segments from the test set of the WMT 2017 - 2019 Metrics Shared Task as provided on the [Unbabel COMET repository](https://github.com/Unbabel/COMET/tree/master/data). As development set, a small portion of the WMT 2020 Metrics Shared Task was used.

The new metrics were evaluated on the `en-de` protion of the official WMT 2020 Metrics Shared Task test set that can be downloaded from [here](https://drive.google.com/drive/folders/1n_alr6WFQZfw4dcAmyxow4V8FC67XD8p). 

The scripts used to prepare the train and dev and sets are included in the folder `data_preparation`. The procedure to create the test set is described [below](#Evaluation). 



## MBR-Decoded Translations
In the first part of the thesis, the translations obtained via MBR decoding with COMET-20 [(Rei et al., 2020)](https://aclanthology.org/2020.emnlp-main.213/) as utility function are analysed and compared to the translations obtained with ChrF and ChrF++ [(Popović, 2015)](https://aclanthology.org/W15-3049/) as utility function. 

To generate the MBR-decoded translations with COMET-20, follow the instructions in [this](https://github.com/chanberg/COMET-mbr) repository by Amrhein and Sennrich (2022).

To generate the MBR-decoded translations with ChrF and ChrF++, follow the instructions in [this](https://github.com/Roxot/mbr-nmt) repository by [Eikema and Aziz (2020)](https://aclanthology.org/2020.coling-main.398/).

The MBR translations with these three utility functions as well as all the other translations obtained with novel utility functions are included in the folder `mbr_translations`. 



## Exploration
The folder `exploration` contains the scripts that were used to semi-automatically analyse the number of unknown words and mistranslated compounds in the MBR-decoded translations. 

First, all unknown words were extracted from the MBR translations. To this end, the translations were compare against the training corpus, the references and the source segment. To reproduce the results, download the [training corpora of the WMT 2018 News Shared Task](https://www.statmt.org/wmt18/translation-task.html) and save them in a folder called `wmt-2018_data`. To analyse the translation e.g. of COMET-20 as utility function run:
```bash
cd exploration
python3 find_unknown_words.py \
		-td en-de \
		-tf ../mbr_translations/comet-20_en-de.txt \
		-o compounds/unknown_words/unknown_comet-20.txt
```

The classify the unknown words into different categories, run: 
```bash
cd exploration
python3 zmorge_socket.py \
		--metric comet-20
```

Then, the lists were manually analysed for mistranslated compounds. To ensure that the results are consistent across the various analysed metrics, `compare_compound_lists.py` compares the unknown words produced by the current metric to the compound lists that have already been analysed. For each unknown word, the script writes to the standard output whether the word has been previously encountered and whether it was classified as mistranslated compound or not (either because it is a correct translation or not a compound at all). When running the script, add all previously collected lists of mistranslated compounds to the `--input` argument and the corresponding lists of unknown words to the `--unknown-words` argument:
```bash
cd exploration
python3 compare_compound_lists.py \
		-i compounds/compounds_mistranslated/mistranslated_compounds_chrf.txt compounds/compounds_mistranslated/mistranslated_compounds_chrf++.txt \
		-c compounds/unknown_words/unknown_comet-20.txt \
		-u compounds/unknown_words/unknown_chrf.txt compounds/unknown_words/unknown_chrf++.txt
```
In the example above, the MBR translations obtained with ChrF and ChrF++ as utility functions were already analysed. The MBR translations of COMET-20 are currently analysed and compared against the previous analyses.



## Training of New Metrics
New metrics were trained to investigate whether Whole Word Masking in pre-training could increase the sensitivity of a metric towards German compounds. 

To this end, the new GBLEURT and GCOMET metrics were trained, both in two flavours: one based on a language model pre-trained with Whole Word Masking (WWM), the other based on a language model pre-trained with Sub-Word Masking (SWM). 
All four metrics build on the German BERT (GBERT) model by [Chan et al. (2020)](
https://aclanthology.org/2020.coling-main.598). The variant pre-trained with WWM is publicly available as on [Hugging Face](https://huggingface.co/deepset/gbert-base) as  `deepset/gbert-base`, while the SWM-based model `gbert-data` is not publicly available. Chan et al. shared it with me upon my request.

The training and [evaluation](#Evaluation) details of [GBLEURT](#GBLEURT) and [GCOMET](#GCOMET) and instructions on how to use them are given below.



## GBLEURT
The training of GBLEURT is implemented from scratch using the [Hugging Face](https://huggingface.co/) library. To train GBLEURT-WWM run:
```bash
cd gbleurt
python3 gbleurt.py --train \
		--gbert_checkpoint deepset/gbert-base \
		--model_name gbleurt-wwm \
		--train_data ../prepared_data/2017-18-19-de-da.csv \
		--dev_data ../prepared_data/2020-de-da-dev.csv		
```
To train GBLEURT-SWM, adjust `--gbert-checkpoint` to `gbert-data` and the `--model_name` to `gbleurt-swm`.

To use GBLEURT for prediction, e.g. on the [WMT 2020 Metrics Shared Task](https://www.statmt.org/wmt20/metrics-task.html) test set: 
```bash
cd gbleurt
python3 gbleurt.py --predict \
		--model_name gbleurt-wwm \
		--test_data ../prepared_data/2020-de-da-official-testset.csv
```

### MBR Decoding with GBLEURT
When implementing MBR decoding with GBLEURT, two different MBR variants are explored. As the candidate pool and the support set are identical for the experiments run in this thesis, each candidate is also contained in the support set. Hence, the comparison of a candidate to itself can either be included (MBR-100) or excluded (MBR-99) in MBR decoding. 

The script can be used with both MBR variants. It will write the chosen translations into an output file. However, as MBR decoding with GBLEURT is costly, the script offers the possibility to write the computed score for each candidate-support pair to a json file. Hence, MBR decoding needs to be run only once with MBR-100. Then, the best candidates according to MBR-100 and MBR-99 can be calculated from the output json file.

To run MBR-100 with GBLEURT-WWM:
```bash
cd gbleurt
python3 run_mbr_gbleurt.py \
		-m gbleurt-wwm \
		-t ../prepared_data/en-de.samples \
		-c ../prepared_data/en-de.samples \
		-o ../mbr_translations/gbleurt-wwm-100_en-de.txt \
		-j ../mbr_translations/gbleurt-wwm_sent-scores.json \
		-ns 100 \
		-nc 100 \
		-ic
```
For GBLEURT-SWM, adjust the model name and the names of the output files accordingly.

Having obtained the json file with the scores for each candidate-support pair, you can find the best candidate for each source segment with:
```bash
cd gbleurt 
python3 mbr_find_best_candidate.py \
		-o ../mbr_translations/gbleurt-wwm-99_en-de.txt \
		-j ../mbr_translations/gbleurt-wwm_sent-scores.json \
```
This command runs MBR-99 decoding. For MBR-100, add the flag `--include_candidate` or `-ic`.



## GCOMET
To train the two GCOMET model as well as COMET-Contrastive the [Unbabel COMET repository](https://github.com/Unbabel/COMET) is used. Specifically, the same version of this repository is used as in [Amrhein and Sennrich (2022)](https://aclanthology.org/2022.aacl-main.83/), where `wmt20-comet-da` is the default model. This version is available [here](https://github.com/chanberg/COMET-mbr). Follow the instructions in these two repositories to install COMET with poetry.

To train the three new metrics, new configuration files are created and saved in the folder `gcomet/configs/models`. The models are trained with:
```bash
cd gcomet
comet-train --cfg configs/models/{your_model_config}.yaml
```
To train GCOMET-WWM use the model config `gcomet-wwm_sourceless_metric.yaml`, GCOMET-SWM use `gcomet-swm_sourceless_metric.yaml` and for COMET-contrastive use `comet-contrastive_sourceless_metric.yaml`

To use GCOMET and COMET-Contrastive for prediction:
```bash
cd gcomet
comet-score -t hyp1.de -r ref.de --model PATH/TO/CHECKPOINT
```

To specifically score the [WMT 2020 Metrics Shared Task](https://www.statmt.org/wmt20/metrics-task.html) test set saving the scores in the format required by the Shared Task:
```bash
cd gcomet
python3 comet_scoring_testset.py \
		--model_name gcomet-wwm \
		--model_path models/gcomet-wwm/epoch=2-step=11130.ckpt \
		--test_data ../prepared_data/2020-de-da-official-testset.csv
``` 
For GCOMET-SWM and COMET-Contrastive, adjust the `model_path` to `models/gcomet-swm/epoch=2-step=11130.ckpt` and `models/comet-contrastive/epoch=4-step=18550.ckpt` respectively and change `model_name` accordingly.


### MBR Decoding with GCOMET and COMET-Contrastive
The MBR decoding is based on the implementation by [Amrhein and Sennrich (2022)](https://github.com/chanberg/COMET-mbr). However, to enable MBR decoding with source-free metrics that take only two input segments (in contrast to COMET-20 that takes three input segments), the script `gcomet/comet/models/regression/referenceless.py` was adjusted accordingly.

To run MBR decoding with GCOMET-WWM: 
```bash
cd gcomet
python3 run_mbr_source-free.py \
		-m models/gcomet-wwm/epoch=2-step=11130.ckpt \
		-t ../prepared_data/en-de.samples \
		-c ../prepared_data/en-de.samples \
		-o ../mbr_translations/gcomet-wwm_en-de.txt \
		-ns 100 \
		-nc 100
```
For GCOMET-SWM and COMET-Contrastive, adjust the model path and output filename accordingly. Change the paths to `models/gcomet-swm/epoch=2-step=11130.ckpt` and `models/comet-contrastive/epoch=4-step=18550.ckpt` respectively. 



## Evaluation
To evaluate the newly trained metrics, the official evalaution from the WMT 2020 Metrics Shared Task is run. The official evaluation scripts are contained in [this repository](https://github.com/WMT-Metrics-task/wmt20-metrics).

To prepare the input files for the evaluation, use the scripts and materials in `wmt20_metrics_evaluation`. 
First, download the newstest2020 data from [here](https://drive.google.com/drive/folders/1n_alr6WFQZfw4dcAmyxow4V8FC67XD8p).

In order to run the official evaluation, the scores assigned by a given metrics have to be saved in the required format. To obtain a file in the required format, run the upper part of the official WMT 2020 script `run_ref_metrics.sh` and write the results to the file `mt_ref_combis.tsv`.

Then, to sort the test set for `en-de` in the required order, run:
```bash
cd wmt20_metrics_evaluation
python3 create_sorted_testset20.py \
		-i ../wmt20/wmt20metrics/newstest2020/txt \
		-d mt_ref_combis.tsv \
		-lp en-de \
		-o ../prepared_data/2020-de-da-official-testset.csv
```

For each metric under study prepare a tab-separated file with the information required by the official evaluation
scripts of the WMT 2020 Metrics Shared Task.

```bash
cd wmt20_metrics_evaluation
python3 create_mt_ref_combis.py \
		-i mt_ref_combis.tsv \
		-d ../wmt20/wmt20metrics/newstest2020/txt/details \
		-lp en-de \
		-m {metric_name} \
		-o results/{metric_name}.seg.score
```

Then, let your metric score the sentences in the `2020-de-da-official-testset.csv` and save the scores in `wmt20_metrics_evaluation/results/{metric_name}.seg.score`. 

To compute the average system-level scores run:
```bash
cd wmt20_metrics_evaluation
python3 compute_sys_scores.py \
		-m {metric_name}
``` 

gzip the file in the `results` folder and copy them to [this folder](https://github.com/WMT-Metrics-task/wmt20-metrics/tree/main/final-metric-scores/submissions) in the official WMT 2020 repository. Then run the [official evaluation script](https://github.com/WMT-Metrics-task/wmt20-metrics/blob/main/results/get-sys-results.sh).



## Combining Metrics in MBR Decoding
To combine COMET-20 and GCOMET-WWM in MBR Decoding, you have to write the scores that the two metrics assign to each candidate to a json file. In a second step, calculate the average between the COMET-20 and the GCOMET-WWM scores and choose the best candidate.

To run MBR decoding with COMET-20 writing the average score per candidate to a json file:
```bash
cd gcomet
python3 run_mbr_comet-20_with-scores.py \
		-m wmt20-comet-da \
		-t ../prepared_data/en-de.samples \
		-s ../prepared_data/en-de.src \
		-c ../prepared_data/en-de.samples \
		-j ../mbr_translations/comet-20_scores.json \
		-ns 100 \
		-nc 100
```

To run MBR decoding with GCOMET-WWM writing the average score per candidate to a json file:
```bash
cd gcomet
python3 run_mbr_source-free_with-scores.py \
		-m models/gcomet-wwm/epoch=2-step=11130.ckpt \
		-t ../prepared_data/en-de.samples \
		-c ../prepared_data/en-de.samples \
		-j ../mbr_translations/gcomet-wwm_scores.json \
		-ns 100 \
		-nc 100
```

Calculate the average between the COMET-20 and the GCOMET-WWM scores and write the candidate with the highest score to the output file:
```bash
cd gcomet
python3 mbr_combine_metrics.py \
		-o ../mbr_translations/combination_comet-20_gcomet-wwm_en-de.txt \
		-j1 ../mbr_translations/comet-20_scores.json \
		-j2 ../mbr_translations/gcomet-wwm_scores.json
```



## MBR-Based Sensitivity Analysis
The sensitivity analysis measures the sensitivity of a metric towards a specific linguistic phenomenon. For this purpose, a challenge set is composed that contains the correct candiate along with perturbed versions of the candidate. In the course of MBR decoding, the metric assigns a score to the correct and the perturbed candidates by comparing them to the two reference translations in the support set. Then, the difference between the score assigned to the correct candidate and the score of the perturbed candidate is calculated. These score differences are then averaged per error type to obtain the final sensitivity score. 

To compose the challenge set for compounds, the 100 candidate translations for each compound under study were collected in a txt file. The following script was run to identify the most frequent incorrect translations of a compound, e.g. `Hundepark`.
```bash
cd sensitivity_analysis
python3 select_compounds_sensitivity.py -n 4 -t hundepark
```


### Sensitivity Analysis with GBLEURT
To run the sensitivity analysis for compounds with GBLEURT:
```bash
cd gbleurt
python3 run_mbr_for_sensitivity_gbleurt.py \
		-m gbleurt-wwm \
		-r ../sensitivity_analysis/challenge_sets/compound-sensitivity_en-de_refs.txt \
		-j ../sensitivity_analysis/challenge_sets/compound-sensitivity_en-de.json \
		-o ../sensitivity_analysis/results/gbleurt-wwm_compound-sensitivity_en-de.json \
		-ns 2
```

To run the sensitivity analysis for nouns, named entities and numbers with GBLEURT: 
```bash
cd gbleurt
python3 run_mbr_for_sensitivity_gbleurt.py \
		-m gbleurt-wwm \
		-r ../sensitivity_analysis/challenge_sets/ne-num-sensitivity_en-de_refs.txt \
		-j ../sensitivity_analysis/challenge_sets/ne-num-sensitivity_en-de.json \
		-o ../sensitivity_analysis/results/gbleurt-wwm_ne-num-sensitivity_en-de.json \
		-ns 2
```
To run the sensitivity analyses for GBLEURT-SWM, adjust the model path to `-m gbleurt-swm`.


### Sensitivity Analysis with GCOMET and COMET-Contrastive
To run the sensitivity analysis for compounds with GCOMET or COMET-Contrastive:
```bash
cd gcomet
python3 run_mbr_for_sensitivity_source-free.py \
		-m models/gcomet-wwm/epoch=2-step=11130.ckpt \
		-r ../sensitivity_analysis/challenge_sets/compound-sensitivity_en-de_refs.txt \
		-j ../sensitivity_analysis/challenge_sets/compound-sensitivity_en-de.json \
		-o ../sensitivity_analysis/results/gcomet-wwm_compound-sensitivity_en-de.json \
		-ns 2
```

To run the sensitivty analysis for nouns, named entities and numbers with GCOMET or COMET-Contrastive: 
```bash
cd gcomet
python3 run_mbr_for_sensitivity_source-free.py \
		-m models/gcomet-wwm/epoch=2-step=11130.ckpt \
		-r ../sensitivity_analysis/challenge_sets/ne-num-sensitivity_en-de_refs.txt \
		-j ../sensitivity_analysis/challenge_sets/ne-num-sensitivity_en-de.json \
		-o ../sensitivity_analysis/results/gcomet-wwm_ne-num-sensitivity_en-de.json \
		-ns 2
```
To run the sensitivity analyses for GCOMET-SWM or COMET-Contrastive, adjust the model path to `-m gcomet-swm/epoch=2-step=11130.ckpt` or `-m comet-contrastive/epoch=4-step=18550.ckpt` respectively.


### Sensitivity Analysis with COMET-20
To run the sensitivity analysis for compounds with COMET-20:
```bash
cd gcomet
python3 run_mbr_for_sensitivity_comet-20.py \
		-m wmt20-comet-da \
		-r ../sensitivity_analysis/challenge_sets/compound-sensitivity_en-de_refs.txt \
		-j ../sensitivity_analysis/challenge_sets/compound-sensitivity_en-de.json \
		-o ../sensitivity_analysis/results/comet-20_compound-sensitivity_en-de.json \
		-ns 2
```

To run the sensitivty analysis for nouns, named entities and numbers with COMET-20: 
```bash
cd gcomet
python3 run_mbr_for_sensitivity_comet-20.py \
		-m wmt20-comet-da \
		-r ../sensitivity_analysis/challenge_sets/ne-num-sensitivity_en-de_refs.txt \
		-j ../sensitivity_analysis/challenge_sets/ne-num-sensitivity_en-de.json \
		-o ../sensitivity_analysis/results/comet-20_ne-num-sensitivity_en-de.json \
		-ns 2
```


### Calculate Sensitivity Scores
To calculate the sensitivity scores for compounds, run:
```bash
cd sensitivity_analysis
python3 evaluate_sensitivity_compounds.py \
		-f results/{metric-name}_compound-sensitivity_en-de.json
```

To calculate the sensitivity scores for nouns, named entities and numbers, run:
```bash
cd sensitivity_analysis
python3 evaluate_sensitivity_ne_num.py \
		-f results/{metric-name}_ne-num-sensitivity_en-de.json
```




