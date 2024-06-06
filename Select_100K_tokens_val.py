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
import jsonlines

from transformers import GPT2TokenizerFast

vectorizer = joblib.load('vectorizer_reddit.joblib')
clf = joblib.load('classifier_reddit.joblib')

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
num_tokens = 0

en = load_dataset("allenai/c4", "en.noclean", split="validation", streaming=True)
i = iter(en)
# start 5:42
with open('c4_reddit_val_100k.txt', 'w') as output_file:
    while (num_tokens < 100000): # start with 100k?? #1M
        # if (num_tokens % 1000 == 0):
        #     print("at" + str(num_tokens))
        text = next(i)["text"]
        #print(text)
        high_qual = score(text, clf, vectorizer)[0][1] > 0.5
        #print(score(text, clf, vectorizer)[0][1])
        tokens_len = len(tokenizer(text)["input_ids"])
        #print(tokens_len)
        if high_qual:
                num_tokens += tokens_len #DOUBLE check
                output_file.write(text + "\n")

print(num_tokens)
