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

print("IN FILE")
# Set these properties for visualization
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("LOADING MODEL")
# Load model
vectorizer = joblib.load('vectorizer_ss.joblib')
clf = joblib.load('classifier_ss.joblib')
print("FINSIHED LOADING MODEL")

wiki_text = """
Blakeney Chapel is a ruined building on the Norfolk coast of England. Even though named as such, it was probably not a chapel, and is not in the adjoining village of Blakeney, but rather in the parish of Cley next the Sea. The building stood on a raised mound or "eye" on the seaward end of the coastal marshes, less than 200 m (220 yd) from the sea and just to the north of the current channel of the River Glaven where it turns to run parallel to the shoreline. It consisted of two rectangular rooms of unequal size, and appears to be intact in a 1586 map, but is shown as ruins in later charts. Only the foundations and part of a wall still remain. Three archaeological investigations between 1998 and 2005 provided more detail of the construction, and showed two distinct periods of active use. Although it is described as a chapel on several maps, there is no documentary or archaeological evidence to suggest that it had any religious function. A small hearth, probably used for smelting iron, is the only evidence of a specific activity on the site.

Much of the structural material was long ago carried off for reuse in buildings in Cley and Blakeney. The surviving ruins are protected as a scheduled monument and Grade II listed building because of their historical importance, but there is no active management. The ever-present threat from the encroaching sea is likely to accelerate following a realignment of the Glaven's course through the marshes, and lead to the loss of the ruins.
"""
print("[prob low qual, prob high qual]")
print(score(wiki_text, clf, vectorizer))

random_webtext = """
As someone who comes from a country with a lot of mountains and hills, I would highly recommend going trekking to places accessible only by foot. It's really nice to see the untouched,peaceful nature existing there, without humans to ruin it.
"""
print("[prob low qual]")
# QUESTION: why is this text considered low qual???
print(score(random_webtext, clf, vectorizer)[0][0])

# for each sample in a sample of dolma
# drop it in a txt file if it categorizes as high quality 
# Load IMDb dataset from Hugging Face

# If the dataset is gated/private, make sure you have run huggingface-cli login
# try ids = ds.to_iterable_dataset(num_shards=64)
dataset = load_dataset("allenai/dolma", split='train', streaming=True, use_auth_token=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)
train_dataset = shuffled_dataset.take(1000000)

# Access the training split
# train_dataset = dataset["train"][:100]
df = pd.DataFrame(columns=["text", "label"])

# Iterate through the dataset line by line
sources = defaultdict(int)
for example in train_dataset:
    # Access each line in the dataset
    text = example["text"]
    source = example["source"]
    # Add to df
    sources[source] +=1 
    df = df.append({"text": text}, ignore_index = True)

scored_df = score_text(df, clf, vectorizer)
token_counts = get_counts(df)
scored_df = pd.concat([scored_df, token_counts['num_tokens']], axis=1)
print("Perc low quality: " + str(len(df[df['prob_low_quality'] > 0.5])/10000))
print("Perc high quality: " + str(len(df[df['prob_high_quality'] > 0.5])/10000))
print("Low qual avg length: " + df[df['prob_low_quality'] > 0.5]["num_tokens"].mean())
#print("High qual avg length: " + str(len(df[df['prob_high_quality'] > 0.5])/10000))
print(sources)

# drop all the ones that GPT3 does NOT include (should just be 0)
    
# def write_to_txt_file():
#     file_name = "dolma_data.txt"
#     with open(file_name, "a", encoding="utf-8") as file:
#         file.write(text)

# df['GPT3_included'].apply(write_to_txt_file)

