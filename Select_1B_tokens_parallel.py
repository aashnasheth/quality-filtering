import joblib
import logging
from data.score import score, score_text, get_counts
import pandas as pd
from datasets import load_dataset

# TODO: check vectorizer, clf, dataset (no-clean), filename
# split we are collecting data from

nt = 0
vectorizer = joblib.load('vectorizer_reddit.joblib')
clf = joblib.load('classifier_reddit.joblib')

def get_high_quality(sample):
    global nt

    with open('c4_reddit_train_1b_1.txt', 'a') as output_file:
        sample_loc = pd.DataFrame(sample)
        sample_loc = get_counts(sample_loc)
        sample_loc = score_text(sample_loc, clf, vectorizer)

        sample_loc = sample_loc.drop(sample_loc[sample_loc.prob_high_quality < 0.5].index)
    
        nt = nt + sample_loc["num_tokens"].sum()
    
        sample_loc["text"].to_csv(output_file, sep='\t',  header=False, index=False, mode='a')
      
        output_file.flush()

        print(nt)
        return sample
                
en = load_dataset("allenai/c4", "en.noclean", streaming=True, split="train")
en_iter = en.map(get_high_quality, batched=True, batch_size=100000)
i = iter(en_iter)

while (nt < 1000000000):
    # next basically calls next batch_size times, and then on the nth time, will perform the map
    # so if batch-size=4 is like: nothing, nothing, nothing, do.... nothing nothing nothing do, but the nothing still logs??
    next(i)
    
