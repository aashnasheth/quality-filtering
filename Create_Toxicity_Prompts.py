#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import evaluate
from statistics import fmean
import numpy as np
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
import jsonlines

en = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
i = iter(en)

toxic_prompts =0
non_toxic_prompts= 0

with jsonlines.open('toxic_prompts.jsonl', mode='w') as writer:
    with jsonlines.open('non_toxic_prompts.jsonl', mode='w') as nt_writer:
        while (toxic_prompts < 1000 or non_toxic_prompts < 1000):
            item = next(i)
            text = item["prompt"]["text"]
            challenging = item["challenging"]
            toxicity = item["prompt"]["toxicity"]
            
            if (toxicity is not None and toxic_prompts < 1000 and toxicity >= 0.5):
                item = {"prompt": {"text" : text}}
                writer.write(item)
                toxic_prompts += 1
            if (toxicity is not None and non_toxic_prompts < 1000 and toxicity < 0.5):
                item = {"prompt": {"text" : text}}
                nt_writer.write(item)
                non_toxic_prompts += 1

print("done!")
    


