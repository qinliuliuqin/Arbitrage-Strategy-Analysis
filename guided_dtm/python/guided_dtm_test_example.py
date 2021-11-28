import time
import pickle
from ldaseqmodel_guided import LdaSeqModel
from gensim import corpora
import numpy as np


def insert_seeds(seeds, beta_s, id2word):
    word2id = id2word.token2id
    for k in range(len(seeds)):
        n = len(seeds[k])
        p = 1.0 / n
        for i in range(n):
            try:
                idx = word2id[seeds[k][i]]
                #print(f"({k},{idx}) {p:.3f}")
                beta_s[idx, k] = np.log(p)
            except Exception as e:
                print(f'ERROR {e}')
            
            

def print_beta(beta, n_words, id2word):
        # Assume beta dim(V,T)
        print(f"beta shape:{beta.shape}")
        t2 = np.transpose(beta)
        K = t2.shape[0]
        N = t2.shape[1]
        for k in range(K):
            print('\nTopic #{}'.format(k))
            i = 0
            a = np.argsort(-t2[k,:])
            for id in a:
                print(f'{id2word[id]} {np.exp(t2[k,id]):.3f}')                
                i += 1
                if i > n_words:
                        break
                        

data = pickle.load( open( "test2.pkl", "rb" ) )

seed_topic_list = [
['loan', 'bond', 'fund', 'model', 'asset']
]

num_topics = 5

print(f"Total numbers of papers:{len(data)}")
print(f"-------Numbers of Topics: {num_topics}--------\n")

x = data
v_dict = corpora.Dictionary(x)
corpus = [v_dict.doc2bow(xi) for xi in x]
print(f'Corpus Length:{len(corpus)}')
vocab_len = len(v_dict)
print(f'Vocabulary Size:{vocab_len}\n')

# Test insert_seeds and print_beta functions
# The values stored in beta is in log domain
V = vocab_len
K = num_topics
log_beta = np.ones((V, K)) * np.log(1e-8) 
insert_seeds(seed_topic_list, log_beta, v_dict)
print_beta(log_beta, 5, v_dict)
np.random.seed(100)

time_slice = [20,20,20,20,13]
pi = 0.5

t1 = time.time()
ldaseq = LdaSeqModel(corpus=corpus, time_slice=time_slice, id2word=v_dict, num_topics=num_topics, chunksize=1, em_min_iter=1, em_max_iter=2, seeds=seed_topic_list, pi=pi)
t2 = time.time()
print(f'The LdaSeqModel took {t2-t1} seconds!')

for t in range(len(time_slice)):
    ldaseq.print_report(t,5)
    x = ldaseq.print_topics(time=t, top_terms=5)
    print(f'\n{x}\n') 
    
  
    
