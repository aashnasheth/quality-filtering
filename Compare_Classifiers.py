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
from data.cluster import cluster_text
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import argparse
parser = argparse.ArgumentParser(description='Type of model to evaluate toxicity on')
parser.add_argument('--ss_thres', action="store")
parser.add_argument('--thres', action="store")
args = parser.parse_args()

print(args.ss_thres)
print(args.thres)
# helpful cluster code: https://jonathansoma.com/lede/algorithms-2017/classes/clustering/k-means-clustering-with-scikit-learn/
def gen_boxplot(df):
    opeds = cluster_text(df,
                     num_clusters=10,
                     num_words=10,
                     plot_boxplot=True,
                     save=True,
                     save_path="ttopic_boxplot.pdf")
    

vectorizer_ss = joblib.load('vectorizer_ss.joblib')
clf_ss = joblib.load('classifier_ss.joblib')

vectorizer = joblib.load('vectorizer.joblib')
clf = joblib.load('classifier.joblib')

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
num_samples = 0

en = load_dataset("allenai/c4", "en.noclean", split="validation", streaming=True)
i = iter(en)

df = pd.DataFrame()
# for 10,000 samples, gather information about length and topic
num_hi_qual_ss = 0
num_hi_qual = 0

num_hi_qual_ss_t = 0
num_hi_qual_t = 0
gen_len = 0
maximum = 0
ss_sample_nums = []
sample_nums = []
gen_len_list = []
while (num_samples < 50000):
    # df2 = pd.DataFrame()
    text = next(i)["text"]
    high_qual_ss = score(text, clf_ss, vectorizer_ss)[0][1]
    high_qual = score(text, clf, vectorizer)[0][1]

    # df2["text"] = text
    # df2["prob_high_quality"] = high_qual
    # df2["prob_high_quality_ss"] = high_qual_ss

    tokens_len = len(tokenizer(text)["input_ids"])
    gen_len += tokens_len
    maximum = max(tokens_len, maximum)
    gen_len_list.append(tokens_len)
    if (high_qual_ss > float(args.ss_thres)):
        num_hi_qual_ss += 1
        num_hi_qual_ss_t += tokens_len
        ss_sample_nums.append(num_samples)
    if (high_qual > float(args.thres)):
        num_hi_qual += 1
        num_hi_qual_t += tokens_len
        sample_nums.append(num_samples)

    df2 = pd.DataFrame([{"text":text, "prob_high_quality":high_qual, "prob_high_quality_ss":high_qual_ss}])
    df = pd.concat([df2, df])
    num_samples+=1

print(df.head(10))
gen_boxplot(df)

print("AVERAGE_TOKEN_LENGTH_SS")
print(num_hi_qual_ss_t/num_hi_qual_ss)
print("AVERAGE_TOKEN_LENGTH")
print(num_hi_qual_t/num_hi_qual)
print("PERCENT CHOSEN SS")
print(num_hi_qual_ss/50000.0)
print("PERCENT CHOSEN")
print(num_hi_qual/50000.0)
print("AVG LEN")
print(gen_len/50000.0)
print("MAX LEN")
print(maximum)

sample_nums = set(sample_nums)
ss_sample_nums = set(ss_sample_nums)
overlap = ss_sample_nums.intersection(sample_nums)
num_overlap= len(overlap)
print("NUM_OVERLAP")
print(num_overlap)

from googleapiclient import discovery
import json
import jsonlines
import time

API_KEY = 'AIzaSyAlejdpfF3IaSo4I0OAYWhZDbbT8VOPxiY'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

request = 0 
with jsonlines.open("perspective_baseline_select_train.jsonl", "w") as writer:
    for item in list(df[df.prob_high_quality >0.5].text)[:10000]:
        print(len(item))
        text = item[:10000]
        print("CROPPED")
        print(len(text))
        try:
            if (request==999):
                time.sleep(60) # wait a minute between batches of requests
                request = 0
            analyze_request = {
            'comment': { 'text': text },
            'requestedAttributes': {'TOXICITY': {}}
            }
            writer.write(analyze_request)
            response = client.comments().analyze(body=analyze_request).execute()
            writer.write({"response":response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]})
            # toxicity_scores.append(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
            request += 1
        except Exception as e:
            print(item) 
            print(e)
            # write 0 if the API can't produce toxicity results to keep numbers even
            writer.write({"response": 0})

request = 0 
with jsonlines.open("perspective_ss_select_train.jsonl", "w") as writer:
    for item in list(df[df.prob_high_quality_ss >0.5].text)[:10000]:
        text = item[:100000]
        try:
            if (request==999):
                time.sleep(60) # wait a minute between batches of requests
                request = 0
            analyze_request = {
            'comment': { 'text': text },
            'requestedAttributes': {'TOXICITY': {}}
            }
            writer.write(analyze_request)
            response = client.comments().analyze(body=analyze_request).execute()
            writer.write({"response":response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]})
            # toxicity_scores.append(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
            request += 1
        except Exception as e:
            print(item) 
            print(e)
            # write 0 if the API can't produce toxicity results to keep numbers even
            writer.write({"response": 0})
# # VenDiagram
# plt.clf()
# venn2([set(ss_sample_nums), set(sample_nums)], set_labels=('SemSchol', 'WikiWebBooks'))
# plt.savefig("ven_diagram.pdf", dpi=300, bbox_inches='tight')

# # box plot
# plt.clf()
# plt.hist(gen_len_list, bins=100,)
# plt.xlabel('Avg Token Length')
# plt.ylabel('Num Samples')
# plt.savefig("token_dist.pdf", dpi=300, bbox_inches='tight')
