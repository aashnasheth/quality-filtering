from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import jsonlines
import transformers
import numpy

from nltk.tokenize import word_tokenize
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Sample documents
# Define and apply stop words removal function
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Remove stop words for each text in the list

samples = 0
num_tokens= 0
corpus = []
with open('filter_data/train.jsonl', 'r') as file:
    for line in file:
        if (num_tokens > 1000000):
            break
        json_data = json.loads(line)
        text = json_data["text"]
        if (json_data["label"]== 1):
            tokenizer_output = len(tokenizer(text)["input_ids"])
            num_tokens += tokenizer_output
            samples += 1
            corpus.append(text)
print(num_tokens/samples)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

corpus = [remove_stopwords(text) for text in corpus]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Step 3: Apply KMeans clustering
num_clusters = 5  # Assuming we want 5 clusters/topics
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Step 4: Identify the top topics
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(num_clusters):
    print(f"Cluster {i+1}:")
    top_words = [terms[ind] for ind in order_centroids[i, :5]]  # Print top 5 words per cluster
    print(", ".join(top_words))

samples = 0
num_tokens= 0
corpus = []
with open('filter_data/train_ss.jsonl', 'r') as file:
    for line in file:
        if (num_tokens > 1000000):
            break
        json_data = json.loads(line)
        text = json_data["text"]
        if (json_data["label"]== 1):
            tokenizer_output = len(tokenizer(text)["input_ids"])
            num_tokens += tokenizer_output
            samples += 1
            corpus.append(text)
print(num_tokens/samples)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

corpus = [remove_stopwords(text) for text in corpus]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Step 3: Apply KMeans clustering
num_clusters = 5  # Assuming we want 5 clusters/topics
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Step 4: Identify the top topics
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(num_clusters):
    print(f"Cluster {i+1}:")
    top_words = [terms[ind] for ind in order_centroids[i, :5]]  # Print top 5 words per cluster
    print(", ".join(top_words))

    