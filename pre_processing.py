from collections import defaultdict
import gensim
from textblob import Word

import warnings

warnings.filterwarnings('ignore')

from nltk.tokenize.casual import casual_tokenize
from nltk.corpus import stopwords
import string
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words

import sys
from unicodedata import category
from nltk.stem import WordNetLemmatizer


# Define the function that removes the references from the papers
def removeRef(document):
    no_ref = document[:document.rfind('References')]
    return no_ref


def root(self):
    word1 = Word(self).lemmatize("n")
    word2 = Word(word1).lemmatize("v")
    word3 = Word(word2).lemmatize("a")
    return word3


# This function will remove all non-vocabulary tokens
def keep_vocab(document):
    r = re.compile("[a-z]+")
    document.sort()
    newlist = list(filter(r.match, document))
    index = document.index(newlist[0])
    del document[0:index]
    return document


# Return the word sequence which only contains the high frequent words defined in
# high_freq_word list
def keep_high_freq(self, high_freq_word):
    words = [word for word in self if word in high_freq_word]
    return words


# Define the function that first tokenized the documents and then remove stopwords and punctuations. This function also perform
# stemmatization and lemmatization.
def extractKeywords(text, punctuation_chars, lemmatizer, stop_words, MyStopword):
    # Remove all numbers from the text
    text_nonum = re.sub(r'\d+', '', text)

    # Remove all email address from the text
    text_nonum = re.sub(r'[\w\.-]+@[\w\.-]+', '', text_nonum)

    # Remove all "Journal of Financial Economics" from the text
    text_nonum = re.sub(r'Journal of Financial Economics', '', text_nonum)

    # Split the text words into tokens
    wordTokens = casual_tokenize(text_nonum)

    # Normalize tokens
    wordTokens = [x.lower() for x in wordTokens]

    # Remove blow punctuation in the list.
    punctuations = [word for word in punctuation_chars]

    # Get all stop words in english.
    stopWords = stop_words

    # Below list comprehension will return only keywords tha are not in stop words and  punctuations
    keywords = [word for word in wordTokens if not word in stopWords and not word in punctuations]

    # Remove words with length<3
    tokens = [word for word in keywords if len(word) > 2]

    # Lemmatization

    token = [lemmatizer.lemmatize(word, 'v') for word in tokens]

    # Convert back to its root

    token = [root(word) for word in token]

    # Reomve the common finance tokens from the documents

    token = [word for word in token if not word in MyStopword]

    return token


###########################################################################
# This function pre-processes the input documents and return the 
# processed texts in a list of list structure.
#  
# Input:  documents  - A list of all documents
#         mode - 0   - Normal processing   
#              - 1   - Restricted outputs,limit the ouput words 
#                      from high frequency words list
#         threshold  - The threshold to determine high frequency words.
#                      Only applicable for mode 1
#
# Return: documents_tokenize - A list of pre-processed documents
############################################################################
def processing(documents, mode=0, threshold=25):
    # Generate stop words list, which will be used in extractKeywords()
    stop_words = stopwords.words('english')
    stop_words[len(stop_words):] = list(sklearn_stop_words)
    
    MyStopword = ['firm', 'return', 'price', 'use', 'market', 'bank', 'value', 'stock']
    stop_words[len(stop_words):] = list(MyStopword)

    # Generate punctuation list, which will be used in extractKeywords()
    punctuation_chars = [chr(i) for i in range(sys.maxunicode)
                         if category(chr(i)).startswith("P")]
    punctuation_chars[len(punctuation_chars):] = string.punctuation

    # Generate lemmatizer,which will be used in function extractKeywords()
    lemmatizer = WordNetLemmatizer()

    # Remove references section from the papers
    documents = [removeRef(x) for x in documents]

    # Clean up text and extract more meaningful words 
    documents_tokenize = [extractKeywords(document, punctuation_chars, lemmatizer, stop_words, MyStopword) 
                          for document in documents]

    # Clean up and sort the document 
    documents_tokenize = [keep_vocab(document) for document in documents_tokenize]

    if (mode != 0):
        # Generate high frequency word(word occurrence>=threshold)dictionary for
        # the whole dataset.    
        frequency = defaultdict(int)
        for document in documents_tokenize:
            for token in document:
                frequency[token] += 1
        high_freq_word = []
        for token in frequency:
            if frequency[token] >= threshold:
                high_freq_word.append(token)

        # Limit the outpur words from the high frequency word dictionary    
        documents_tokenize = [keep_high_freq(document, high_freq_word) for document in documents_tokenize]

    return documents_tokenize
