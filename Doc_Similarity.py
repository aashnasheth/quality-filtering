from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import jsonlines
import transformers
import numpy

from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Sample documents

doc1 = ""
num_tokens= 0
with open('filter_data/train.jsonl', 'r') as file:
    for line in file:
        if (num_tokens > 100000):
            break
        json_data = json.loads(line)
        text = json_data["text"]

        if (json_data["label"]== 1):
            tokenizer_output = len(tokenizer(text)["input_ids"])
            num_tokens += tokenizer_output
            doc1 += text

doc2 = ""
num_tokens= 0
with open("paloma_4chan_val_100k.txt", 'r') as file:
    while (num_tokens < 100000):
        text = file.readline()
        tokens_len = len(tokenizer(text)["input_ids"])
        num_tokens += tokens_len #DOUBLE check
        doc2 += text

# print(doc2)
# Vectorize the documents

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

# Calculate cosine similarity

# measure of overlaping vocab essentially
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
print(f"Cosine Similarity: {cosine_sim[0][1]}")

# https://spotintelligence.com/2022/12/19/text-similarity-python/