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

en = load_dataset("allenai/dolma", split="train", streaming=True)
i = iter(en)
# start 5:42
with open('dolma_val_100k.txt', 'w') as output_file:
    #while (num_tokens < 100000): # start with 100k?? #1M
        # if (num_tokens % 1000 == 0):
        #     print("at" + str(num_tokens))
        while True:
            if (num_tokens >= 100000):
                   break
            item = next(i)
            text = item["text"]
            source = item["source"]
            #print(tokens_len)
            if source != "c4" and source != "cc":
                    tokens_len = len(tokenizer(text)["input_ids"])
                    num_tokens += tokens_len #DOUBLE check
                    output_file.write(text + "\n")
            

print(num_tokens)
# with open("cc_en_head-0000.json", 'r') as file:
#     with open('1b_tokens.txt', 'w') as output_file:
#         for line in file:
#             # Access each line in the dataset
#             if (num_tokens > 1000000000):
#                 break
            
#             json_data = json.loads(line)
#             text = json_data["text"]

#             tokenizer_output = tokenizer(text)["input_ids"]
#             if (len(tokenizer_output) > 0): # skip empty lines
#                 high_qual = score(text, clf, vectorizer)[0][1] > 0.5
#                 if high_qual:
#                         num_tokens += tokenizer_output[0] #DOUBLE check
#                         output_file.write(text + "\n")

# # started at: 12:39, end 1:10 30 MIN
# # TODO: double check                    
# print(num_tokens)

