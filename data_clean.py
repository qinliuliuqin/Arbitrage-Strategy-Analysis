#%%
import pickle
import os
import docx
from collections import defaultdict
import gensim
from textblob import Word
import pandas as pd
import random
import numpy as np
from gensim.models import TfidfModel
from textblob import Word
from nltk.tokenize.casual import casual_tokenize
from nltk.corpus import stopwords
import string
import re
from itertools import compress
#%%

#%%
pkl_file_path = 'C:/Users/brave/OneDrive/Desktop/Summer 2021/data.pkl'
with open(pkl_file_path, 'rb') as f:
    docs = pickle.load(f, encoding='bytes')
#%%

#%%
#Define the function that removes the references from the papers
def removeRef(document):
    no_ref=document[:document.find('References')]
    
    return no_ref
#%%

#%%
documents = []
for journal in docs:
    for year in docs[journal]:
        for month in docs[journal][year]:
            for paper in docs[journal][year][month]:
                documents.append(paper)
#%%

#%%
documents=[removeRef(x) for x in documents]
#documents =[removeTitle(x) for x in documents]
#%%

#%%
def clean_data(text):
    text = re.sub(r'-\n\s', '',text)
    text = re.sub(r'-\n','',text)
    text = re.sub(r'[0-9]+','',text)
    text = re.sub(r'The Review of Financial Studies','',text)
    text = re.sub(r'The Journal of Finance','',text)
    text = re.sub(r'Journal of Financial Economics','',text)
    text = re.sub(r'This content downloaded from \n�������������... on .*?,  .*?  :: UTC������������� \n\nAll use subject to https://about.jstor.org/terms','',text)
    text = re.sub(r'%s(.+?)%s'%('\n\n\n\n\n\n ','\n\n '),'',text)
    text = re.sub(r'%s(.+?)%s'%('\n\n ','\n\n '),'',text)
    #text = re.sub(r'[^A-Za-z0-9 -]+','',text)
    return text
#%%

#%%
def clean_str(text):
    text = re.sub(r'\b\w{{{}}}\b'.format(1), '',text)
    text = re.sub(r'\b\w{{{}}}\b'.format(2), '',text)
    
    return text
#%%


#%%
def re_join(token):
    texts = (" ").join(token)
    
    return texts
#%%

#%%
documents = [clean_data(x) for x in documents]
#%%


#%%
import texthero as hero
#%%

#%%
text=pd.DataFrame({'documents':documents})
#%%

#%%
documents_clean = hero.remove_html_tags(text.documents)
documents_clean = hero.clean(documents_clean)
documents_clean=documents_clean.apply(clean_str)
documents_clean = hero.remove_whitespace(documents_clean)
#%%

#%%
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
bow_model = CountVectorizer(tokenizer=casual_tokenize,ngram_range=(1,2))
bow_docs = bow_model.fit_transform(raw_documents=documents_clean)
#%%

#%%
def bigram_frequency(model,docs,threshold):
    L = int(np.floor(docs.shape[1]/100000))
    bigram_feature_name=model.get_feature_names()
    high_freq_index = np.array(True)
    for i in range(1,L):
        tfidf_matrix = docs[:,((i-1)*100000):(i*100000)].toarray()
        matrix = tfidf_matrix!=0
        high_freq = np.sum(matrix,axis=0) > threshold
        high_freq_index = np.append(high_freq_index,high_freq)
    tfidf_matrix = docs[:,(L*100000):].toarray()
    matrix = tfidf_matrix!= 0
    high_freq = np.sum(matrix,axis=0) > threshold
    high_freq_index = np.append(high_freq_index,high_freq)
    high_freq_index = high_freq_index[1:]
    bigram_matrix = docs[:,high_freq_index].toarray()
    bigram_vocabulary = list(compress(bigram_feature_name, high_freq_index))
    
    return bigram_matrix,bigram_vocabulary
#%%

#%%
dtm_matrix,dtm_vocabulary = bigram_frequency(bow_model, bow_docs, threshold=62)
#%%

#%%
from lda import guidedlda as glda
dictionary = dict(zip(dtm_vocabulary, list(range(len(dtm_vocabulary)))))
#%%

#%%
seed_topic_list = [['book market', 'book tomarket'],
                   ['earnings ratio','earnings ratios'],
                   ['earnings surprise', 'earnings surprises'],
                   ['capm beta','capm betas','beta','beta asset']]
#%%

#%%
model = glda.GuidedLDA(n_topics=100, n_iter=100, random_state=7, refresh=20)
#%%

#%%
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[dictionary[word]] = t_id
#%%

#%%
model.fit(dtm_matrix, seed_topics=seed_topics, seed_confidence=0.15)
#%%


