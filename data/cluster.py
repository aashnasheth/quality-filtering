from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from data.viz import boxplot


sns.set(style='white', font_scale=1.4, context='paper')

def cluster_text(z, num_clusters=10, num_words=10, plot_boxplot=False, save=False, save_path=None):
    cv = TfidfVectorizer(
        max_features=10000,
        min_df=3,
        stop_words="english")
    vecs = cv.fit_transform(z.text.str.lower())
    svd = TruncatedSVD(100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    vecs = lsa.fit_transform(vecs)
    km = KMeans(n_clusters=num_clusters)
    km.fit(vecs)
    z['cluster'] = km.labels_
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    z = z.reset_index()
    print(z.head(10))
    terms = cv.get_feature_names_out()
    clusters = {"cluster": [], 
                         "top_words":  [],
                         "quality":  [],
                         "probs": []}
    ss_clusters = {"cluster": [], 
                         "top_words":  [],
                         "quality":  [],
                         "probs": []}
    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :num_words]]
       
        for item in z.loc[z.cluster == i].prob_high_quality:
            clusters["cluster"].append(i)
            print(i)
            print(" ".join(top_words))
            clusters["top_words"].append(i)
            clusters["quality"].append("WikiWebBooks")
            clusters["probs"].append(item)

        for item in z.loc[z.cluster == i].prob_high_quality_ss:
            ss_clusters["cluster"].append(i)
            ss_clusters["top_words"].append(i)
            ss_clusters["quality"].append("SemanticScholar")
            ss_clusters["probs"].append(item)
        # clusters.append({"cluster": i, 
        #                  "top_words": " ".join(top_words),
        #                  "quality": "WikiWebBooks",
        #                  "probs": z.loc[z.cluster == i].prob_high_quality.mean()})
        # clusters.append({"cluster": i, 
        #                  "top_words": " ".join(top_words),
        #                  "quality": "SemanticScholar",
        #                  "probs": z.loc[z.cluster == i].prob_high_quality_ss.mean()})

    data = {"cluster": clusters["cluster"], 
                         "top_words":  clusters["top_words"],
                         "quality":  clusters["quality"],
                         "probs": clusters["probs"]}
    ss_data = {"cluster": ss_clusters["cluster"], 
                         "top_words":  ss_clusters["top_words"],
                         "quality":  ss_clusters["quality"],
                         "probs": ss_clusters["probs"]}
    df = pd.DataFrame(data)
    ss_df = pd.DataFrame(ss_data)

    plt.clf()
    #sns.boxplot(data=df, y='top_words', x='probs', hue='quality', orient='h', linewidth=2, showfliers=False, gap=0.2)
    medians = df.groupby('top_words')['probs'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, y='top_words', x='probs', orient='h', linewidth=2, showfliers=False,color='yellow', order=medians)
    plt.xlabel('Quality')
    plt.ylabel('Cluster Top Words')
    plt.legend(title='WikiWebBooks', loc='upper right')
    plt.savefig("topic_box_plot.pdf", dpi=300, bbox_inches='tight')

    plt.clf()
    medians = ss_df.groupby('top_words')['probs'].median().sort_values(ascending=False).index
    sns.boxplot(data=ss_df, y='top_words', x='probs', orient='h', linewidth=2, showfliers=False, color='orange', order=medians)
    plt.xlabel('Quality')
    plt.ylabel('Cluster Top Words')
    plt.legend(title='SemanticScholar', loc='upper right')
    plt.savefig("topic_box_plot_ss.pdf", dpi=300, bbox_inches='tight')
    # clusters = pd.DataFrame(clusters)
    # print(clusters.head(10))
    # z['cluster'] = z.cluster.astype('category')
    # if plot_boxplot:
    #     boxplot(z, clusters, save=save, save_path=save_path)
    return z

