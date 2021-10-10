from sklearn.cluster import MiniBatchKMeans
import numpy as np

from pre_processing import processing
from file_helper import load_data

import time
import gensim.downloader as api

# Function to convert document(Paper)tokens to vectors by using 
# word2vec word embedding. 
def doc2vec(doc, model):
    vecs = []
    for token in doc:
        if token in model:
            try:
                vecs.append(model[token])
            except KeyError:
                continue
    np_vectors = np.asarray(vecs)
    
    return np_vectors

# K-mean clustering by using sklearn.cluster.MiniBatchKMeans
def k_mean_clustering(x, k=10, i=100, s=1024):
    km = MiniBatchKMeans(n_clusters=k, max_iter=i, verbose=0, random_state=0, batch_size=s)
    c = km.fit(x)
    return c


# Load papers from binary files
print('Loading Papers')
start_time = time.time()
docs= load_data('data.pkl')

documents = []
journal = 'JFE'
year = '2001'
months = ['01', '02', '03']

i = 0
for month in months:
    for paper in docs[journal][year][month]:
            documents.append(paper)
            i += 1
print('Total Papers:%d\nTime:%f' % (i, (time.time() - start_time)))

# Pre-process papers
print('Pre-processing documents start')
start_time = time.time()
documents_tokenize = processing(documents, mode=1, threshold=25)
print('Pre-processing documents ends.\nTime:%f' % (time.time() - start_time))


#print(len(documents_tokenize))

# load pre trained word2vec model
print('Loading pre-trained word2vec model')
start_time = time.time()
w2v_model = api.load('word2vec-google-news-300')    
print('Loading word2vec model took %f seconds' % (time.time() - start_time))

# Convert document to vectors 
doc_list = []
i = 0
print('word2vec convertion starts')
start_time = time.time()
for document in documents_tokenize:
    #print('Document:%d' % i)
    x = doc2vec(document, w2v_model)
    #print(x.shape)
    doc_list.append(x)
    i += 1
print('word2vec convertion ends.\nTime:%f' % (time.time() - start_time))

# Clustering with K mean
# n_k: Numbers of clusters
n_k =  10
print('K mean clustering starts')
start_time = time.time()
clusters= []
for doc in doc_list:
    c = k_mean_clustering(doc, k=n_k)
    clusters.append(c)
print('K mean clustering ends.\nTime:%f' % (time.time() - start_time))

# Print cluster results
print('Print clustering information starts')
i = 0
start_time = time.time()
for c in clusters:
    print('DOC#%d' % i)
    i += 1
    print("Most representative terms per cluster (based on centroids):")
    for j in range(n_k):
        tokens_per_cluster = ""
        most_representative = w2v_model.most_similar(positive=[c.cluster_centers_[j]], topn=5)
        for t in most_representative:
            tokens_per_cluster += f"{t[0]} "
        print(f"Cluster {j}: {tokens_per_cluster}")
print('Print clustering information ends\nTime:%f' % (time.time() - start_time))



