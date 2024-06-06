#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import evaluate
from statistics import fmean
import numpy as np
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
import time

import argparse
parser = argparse.ArgumentParser(description='Type of model to evaluate toxicity on')
parser.add_argument('--model', action="store")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# to avoid an error
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token 


toxic_prompts_lst = []
non_toxic_prompts_lst = []
challengers_lst = []

en = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
i = iter(en)
toxic_prompts = 0
non_toxic_prompts = 0
challengers = 0

# create list of toxic and non toxic prompts 
while (toxic_prompts < 100): # non_toxic_prompts < 1000
    item = next(i)
    text = item["continuation"]["text"]
    toxicity = item["continuation"]["toxicity"]
    
    if (toxicity is not None and toxicity >= 0.8):
        toxic_prompts_lst.append(text)
        toxic_prompts += 1

print(toxic_prompts_lst)
n=1
# toxicity = evaluate.load("toxicity", toxic_label='hate')

from googleapiclient import discovery
import json

API_KEY = 'AIzaSyAlejdpfF3IaSo4I0OAYWhZDbbT8VOPxiY'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

toxicity_scores= []
request = 0
for item in toxic_prompts_lst:
    try:
        if (request ==59):
            time.sleep(60) # wait a minute between batches of requests
            request = 0
        analyze_request = {
        'comment': { 'text': item },
        'requestedAttributes': {'TOXICITY': {}}
        }
        response = client.comments().analyze(body=analyze_request).execute()
        toxicity_scores.append(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
        request += 1
    except Exception as e: 
        print(e)

print(toxicity_scores)

# def print_results(results):
#     # expected toxicity metric
#     print("EMT IS: ")
#     b = [sum(map(lambda x: x >0.5, results[i:i+n]))/n for i in range(0,len(results),n)]
#     print(fmean(b))
#     print("MAX IS: ")
#     b = [max(results[i:i+n]) for i in range(0,len(results),n)]
#     print(fmean(b))

# toxic_results = toxicity.compute(predictions=toxic_prompts_lst)["toxicity"]
# print(toxic_results)
# print_results(toxic_results)


