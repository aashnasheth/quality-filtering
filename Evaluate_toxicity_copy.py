#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import evaluate
from statistics import fmean
import numpy as np
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
import time
import jsonlines
import json

import argparse
parser = argparse.ArgumentParser(description='Type of model to evaluate toxicity on')
parser.add_argument('--model', action="store")
parser.add_argument('--output', action="store")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model)

# ss_model = AutoModelForCausalLM.from_pretrained("./tmp/ss_model/checkpoint-3648")
# baseline_model = AutoModelForCausalLM.from_pretrained("./tmp/baseline_model/checkpoint-3648")
# gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",  pad_token="<|endoftext|>")
# to avoid an error
# tokenizer.padding_side = "left" 
# tokenizer.pad_token = tokenizer.eos_token 


toxic_prompts_lst = []

with open("toxic_prompts.jsonl", "r") as file:
    for line in file:
        data = json.loads(line)
        prompt_text = data["prompt"]["text"]
        toxic_prompts_lst.append(prompt_text)
   
n = 25
toxic_prompts_lst = np.repeat(toxic_prompts_lst, n).tolist()
encoding = tokenizer(toxic_prompts_lst, return_tensors='pt', pad_to_max_length=True)
generated_ids = model.generate(**encoding, max_new_tokens=20, do_sample=True, top_p=0.9)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

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

request = 0 
with jsonlines.open(args.output, "w") as writer:
    for item in generated_texts:
        try:
            if (request==999):
                time.sleep(60) # wait a minute between batches of requests
                request = 0
            analyze_request = {
            'comment': { 'text': item },
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

# print_results(toxicity_scores)
    