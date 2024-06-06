from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import jsonlines
import time

from nltk.tokenize import word_tokenize
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

samples = 0
num_tokens= 0
corpus = []
with open('filter_data/train_ss.jsonl', 'r') as file:
    for line in file:
        if (num_tokens > 100000):
            break
        json_data = json.loads(line)
        text = json_data["text"]
        if (json_data["label"]== 1):
            tokenizer_output = len(tokenizer(text)["input_ids"])
            num_tokens += tokenizer_output
            samples += 1
            corpus.append(text)
print(num_tokens/samples)

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
with jsonlines.open("perspective_baseline_train.jsonl", "w") as writer:
    for item in corpus:
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


    