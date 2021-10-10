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
tfidf_model = CountVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=text.documents).toarray()
#%%


