import jsonlines
from datasets import load_dataset
import joblib
from lr.train import train_lr
from lr.hyperparameters import BEST_HPS
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
import datetime

# edit the load_dataset to pick from the dataset we want

# SEM SCHOL
# en = load_dataset("allenai/c4", "en.noclean", split="train", streaming=True)
# REDDIT
# https://huggingface.co/datasets/SophieTr/reddit_clean

# edit the output file paths
input_files = ["filter_data/dev.jsonl", "filter_data/test.jsonl", "filter_data/train.jsonl"]
output_files = ["filter_data/dev_reddit.jsonl", "filter_data/test_reddit.jsonl", "filter_data/train_reddit.jsonl"]

with jsonlines.open("reddit-0000.json", 'r') as reddit_reader:
    for input_file,output_file in zip(input_files, output_files):
            with jsonlines.open(input_file, 'r') as reader:
                with jsonlines.open(output_file, 'w') as writer:
                    lines = list(reader)
                    for line in lines:
                        # replace what we consider high quality with our data
                        if (line.get('label') == 1):
                            # while (en[idx]["source"] != "reddit"):
                            #     print(en[idx]["source"])
                            #     idx+=1
                            line['text'] = reddit_reader.read()["text"] 
                        writer.write(line)

# Read data into memory
# edit these filepaths
print("READING DATA")
train = pd.read_json('filter_data/train_reddit.jsonl', lines=True)
dev = pd.read_json('filter_data/dev_reddit.jsonl', lines=True)
test = pd.read_json('filter_data/test_reddit.jsonl', lines=True)
print("FINISHED READING DATA")

## Train Classifer
clf, vectorizer, results = train_lr(train, dev, test, BEST_HPS)
print(results)

# edit these filepaths
joblib.dump(vectorizer, 'vectorizer_reddit.joblib')
joblib.dump(clf, 'classifier_reddit.joblib')
