#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import numpy as np
import joblib
from lr.train import train_lr
from data.score import score, score_text, get_counts
from lr.eval import load_model
from lr.hyperparameters import BEST_HPS
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
import datetime
import os
import json

from transformers import GPT2TokenizerFast

# TODO: check vectorizer, clf, dataset (no-clean), filename
# split we are collecting data from

vectorizer = joblib.load('vectorizer_reddit.joblib')
clf = joblib.load('classifier_reddit.joblib')

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
num_tokens = 0
counter = 0

en = load_dataset("allenai/c4", "en.noclean", split="train", streaming=True)
i = iter(en)

# naming schema: dataset_classifiername_trainorval_size
with open('train_dataset_log_reddit.txt', 'w') as print_file:
    with open('c4_reddit_train_1b.txt', 'w') as output_file:
        while (num_tokens < 300000000):
            if (counter > 100000): # print every 100k tokens
                 counter = 0
                 print_file.write("at " + str(num_tokens) + "\n")
            text = next(i)["text"]
           
            high_qual = score(text, clf, vectorizer)[0][1] > 0.5
         
            tokens_len = len(tokenizer(text)["input_ids"])
         
            if high_qual:
                    num_tokens += tokens_len
                    counter += tokens_len
                    output_file.write(text + "\n")
            print_file.flush()
