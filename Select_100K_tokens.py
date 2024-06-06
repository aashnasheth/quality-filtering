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

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# num_tokens = 0
# en = load_dataset("allenai/peS2o", split="train", streaming=True)
# i = iter(en)
# with open('pes2o_val_100k.txt', 'w') as output_file:
#     while (num_tokens < 100000):
#         text = next(i)["text"]
#         tokens_len = len(tokenizer(text)["input_ids"])
#         num_tokens += tokens_len #DOUBLE check
#         output_file.write(text + "\n")

# print(num_tokens)

# num_tokens = 0
# with jsonlines.open("reddit-0000.json", 'r') as reddit_reader:
#     with open('reddit_val_100k.txt', 'w') as output_file:
#         while (num_tokens < 100000):
#             text = reddit_reader.read()["text"]
#             tokens_len = len(tokenizer(text)["input_ids"])
#             num_tokens += tokens_len #DOUBLE check
#             output_file.write(text + "\n")

# print(num_tokens)

# num_tokens = 0
# with jsonlines.open("wiki-0000.json", 'r') as reddit_reader:
#     with open('wiki_val_100k.txt', 'w') as output_file:
#         while (num_tokens < 100000):
#             text = reddit_reader.read()["text"]
#             tokens_len = len(tokenizer(text)["input_ids"])
#             num_tokens += tokens_len #DOUBLE check
#             output_file.write(text + "\n")

# print(num_tokens)

num_tokens = 0
with jsonlines.open("../books-0000.json", 'r') as reddit_reader:
    with open('books_val_100k.txt', 'w') as output_file:
        while (num_tokens < 100000):
            text = reddit_reader.read()["text"]
            tokens_len = len(tokenizer(text)["input_ids"])
            num_tokens += tokens_len #DOUBLE check
            output_file.write(text + "\n")

print(num_tokens)
