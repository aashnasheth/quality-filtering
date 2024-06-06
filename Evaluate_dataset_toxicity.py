#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import evaluate
from statistics import fmean
import numpy as np
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
import json

samples = []
num_sample = 0
with open("filter_data/train.jsonl", 'r') as file:
    for line in file:
        if (num_sample >= 1000):
            break
        json_data = json.loads(line)
        text = json_data["text"]
        if (json_data["label"]==1):
            samples.append(text)
            num_sample +=1
print(num_sample)

toxicity = evaluate.load("toxicity", module_type="measurement")
toxic_results = toxicity.compute(predictions=samples)["toxicity"]
print(fmean(toxic_results))




