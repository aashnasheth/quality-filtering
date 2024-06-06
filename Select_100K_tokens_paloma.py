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

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
num_tokens = 0

files = ["val.jsonl"] # , "gab/val-00000001.jsonl"
with open('wiki_val_200k.txt', 'w') as output_file:
    file_index = 0
    while (num_tokens < 200000):
        print(num_tokens)
        print("reading " + files[file_index])
        with open(files[file_index], 'r') as file:
            for line in file:
                if (num_tokens > 200000):
                    break
                json_data = json.loads(line)
                text = json_data["text"]

                tokenizer_output = len(tokenizer(text)["input_ids"])
                num_tokens += tokenizer_output
                output_file.write(text + "\n")
        file_index+=1
print(num_tokens)

