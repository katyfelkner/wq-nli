# WinoQueer-NLI Bias Benchmark

## Overview
This repository contains the WinoQueer-NLI benchmark, a dataset for measuring homophobic and transphobic bias in language models using a textual entailment task. It is based on the original [WinoQueer dataset](https://github.com/katyfelkner/winoqueer), which uses a token probability evaluation metric.

## Paper
Link will be provided when available.

## Repo Contents
* code/train_mnli.py - script we used to do task finetuning on MNLI dataset of off-the-shelf and previously debiased models
* code/eval_finetuned_mnli.py - script we used to check MNLI performance of our finetuned models
* code/eval_wqnli.py - script to evaluate a model on the WQNLI dataset. **If you want to evaluate your own models on this dataset, use this!**
* code/Make NLI dataset.ipynb - details of how we created WQ-NLI from the original WQ dataset.
* code/metadata matching.ipynb - how we matched WQ-NLI instances with additional metadata to allow fine-grained analysis
* code/other notebooks: data analysis and figures for our paper
* data/winoqueer_nli.csv - the official WQ-NLI dataset. **Use this to evaluate your models!**
* data/datasets_with_metadata/ - WQ and WQ-NLI datasets with additional metadata to allow fine-grained analysis
* data/wq_nli_results/ - raw results from WQ-NLI evaluation
* data/wq_tp_results/ - raw results from WQ-TP evaluation
* data/templates/ - used in WQ-NLI dataset creation
* data/extreme_examples/ - highest and lowest scoring benchmark sentences for each model variant