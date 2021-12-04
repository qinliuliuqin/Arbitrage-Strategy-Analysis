import os
from scipy.special import psi, polygamma, gammaln
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import pickle
import sys

# Utility functions
def dg(gamma, d, i):
    """
    E[log θ_t] where θ_t ~ Dir(gamma)
    """
    return psi(gamma[d, i]) - psi(np.sum(gamma[d, :]))


def dl(lam, i, w_n):
    """
    E[log β_t] where β_t ~ Dir(lam)
    """
    return psi(lam[i, w_n]) - psi(np.sum(lam[i, :]))




def n_most_important(beta_i, n=30):
    """
    find the index of the largest `n` values in a list
    """
    
    max_values = beta_i.argsort()[-n:][::-1]
    return np.array(my_vocab)[max_values]


def print_topic(t_w_dist, top_terms, id2word):
    #t2 = np.transpose(t_w_dist)
    t2 = t_w_dist
    K = t2.shape[0]
    N = t2.shape[1]
    for k in range(K):
        print('\nTopic #{}'.format(k))
        i = 0
        a = np.argsort(-t2[k,:])
        for id in a:
            print('{} {:.3f}'.format(id2word[id],t2[k,id]))
            i += 1
            if i > top_terms:
                    break


def print_topic_s(t_s, terms, id2word):
    for k in range(len(t_s)):
        print('\nTopic #{}'.format(k))
        i = 0
        v = t_s[k]
        print(v)
        print(v.shape)
        a = np.argsort(-v[:,0])
        #print(a)
        for id in a:
            print('{} {:.3f}'.format(id2word[id],v[id,0]))
            i += 1
            if i > 5:
                    break

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

#def sigmoid(x):
#    return 1/(1 + np.exp(-x))


class GuidedLda:

    def __init__(self, docs, vocab, n_topic):
        #global V, k, N, M, alpha, beta, gamma, phi, ss, beta_s, eta
        self.docs = docs
        V = len(vocab)
        K = n_topic  # number of topics
        N = np.array([doc.shape[0] for doc in docs])
        self.n_sum = np.sum(N)
        self.n_max = max(N)
        M = len(docs)
        self.V = V
        self.K = K
        self.N = N
        self.M = M

        print(f"V: {V}\nK: {K}\nN: {N[:10]}...\nM: {M}")

        # initialize α, β

        self.alpha = np.random.gamma(shape=100, scale=0.01, size=K)  # np.random.rand(K)
        #alpha = np.ones(K) * 0.01
        print(f"Init alpha:{self.alpha} shape:{self.alpha.shape}")
       
        self.beta = np.random.dirichlet(np.ones(V), K)
        #print(f"Init beta[1]:{beta[1, :]}")
        print(f"α: dim {self.alpha.shape}\nβ: dim {self.beta.shape}")

        # initialize ϕ, γ
        ## ϕ: (M x max(N) x K) arrays with zero paddings on the right
        self.gamma = self.alpha + np.ones((M, K)) * N.reshape(-1, 1) / K

        self.phi = np.ones((M, max(N), K)) / K
        for m, N_d in enumerate(N):
            self.phi[m, N_d:, :] = 0  # zero padding for vectorized operations

        self.eta = np.ones((M, max(N))) * 0.5


        #ss = np.zeros((V, K))
        #beta_s = np.zeros((V, K))

        print('phi: dim {}'.format(self.phi.shape))
        print(f"γ: dim {self.gamma.shape}\nϕ: dim ({len(self.phi)}, N_d, {self.phi[0].shape[1]})")
        print(f"η: dim {self.eta.shape}\n")
    

    def E_step(self):
        """
        Minorize the joint likelihood function via variational inference.
        This is the E-step of variational EM algorithm for LDA.
        """
        # optimize phi
        for m in range(self.M):
            Nm = self.N[m]
            # gamma[m,:] = alpha + phi[m,:,:].sum(axis=0)
            # t0 dim(T,1)
            t0 = np.exp(psi(self.gamma[m, :]) - psi(self.gamma[m, :].sum())).reshape(-1, 1)
            # word indexes in docs[m], length Nm
            idx = self.docs[m]
            # t1 dim(T,Nm)
            t1 = self.beta[:, idx]
            # t0 * t1  dim(T, Nd), t2 dim(Nm, T)
            t2 = (t0 * t1).T
            # phi[m, :Nm, :] dim(Nm, T)
            self.phi[m, :Nm, :] = t2

            # Normalize phi
            # phi[m, :Nm] dim(Nm,T), phi[m, :Nm].sum(axis=1) dim(Nm,), reshape dim(Nm,1)
            # dim(Nm, T) / dim(N,1)
            self.phi[m, :Nm] /= self.phi[m, :Nm].sum(axis=1).reshape(-1, 1)
            if np.any(np.isnan(self.phi)):
                raise ValueError("phi nan")
            # gamma[m, :] = alpha + phi[m, :, :].sum(axis=0)

            # Update sufficient statistics
            # ss[docs[m],:] dim(Nm,T), phi[m,:,:] dim(Nm,T)
            #ss[docs[m], :] += phi[m, :Nm, :]

            #for j in docs[m]:
                #ss[j, :] += (docs[m] == j) @ phi[m, :Nm, :]

        # optimize gamma
        # gamma dim(M,T)  alpha dim(T,) phi.sum(axis=1) dim(M,T)
        self.gamma = self.alpha + self.phi.sum(axis=1)
    

    def E_step_s(self, beta_r, beta_s):
        """
        Minorize the joint likelihood function via variational inference.
        This is the E-step of variational EM algorithm for LDA.
        """
        # optimize phi
        e = 1e-10
        for m in range(self.M):
            Nm = self.N[m]

           #  # Update eta
           #  idx = self.docs[m].astype(int)
           #  # t1 dim(T,Nm)
           #  t1 = np.log(beta_s[:, idx]+e) + c_log_pi - np.log(beta_r[:, idx]+e) - c_log1_pi
           #  # t2 dim(Nm, T)
           #  t2 = self.phi[m, :Nm, :]
           #  # t3 dim(T, Nm)
           #  t3 = t1 * t2.T
           #  # t4 dim(Nm,)
           #  t4 = t3.sum(axis=0)
           #  # dim(Nm,)
           # self.eta[m, :Nm] = sigmoid(t4)

            # gamma[m,:] = alpha + phi[m,:,:].sum(axis=0)
            # t0 dim(T,1)
            t0 = (psi(self.gamma[m, :]) - psi(self.gamma[m, :].sum())).reshape(-1, 1)
            # word indexes in docs[m], length Nm
            idx = self.docs[m].astype(int)
            # t1 dim(T,Nm) eta dim (1,Nm)
            t1_s = self.eta[m, :Nm] * np.log(beta_s[:, idx] * pi +e)
            t1_r = (1 - self.eta[m, :Nm]) * np.log(beta_r[:, idx] * (1-pi) +e)
            # dim(T, Nm)
            t1 = t1_r + t1_s
            # t0 + t1  dim(T, Nm), t2 dim(Nm, T)
            t2 = (t0 + t1).T
            # phi[m, :Nm, :] dim(Nm, T)
            self.phi[m, :Nm, :] = np.exp(t2)


            # Normalize phi
            # phi[m, :Nm] dim(Nm,T), phi[m, :Nm].sum(axis=1) dim(Nm,), reshape dim(Nm,1)
            # dim(Nm, T) / dim(N,1)
            self.phi[m, :Nm] /= self.phi[m, :Nm].sum(axis=1).reshape(-1, 1)
            if np.any(np.isnan(self.phi)):
                raise ValueError("phi nan")
            # gamma[m, :] = alpha + phi[m, :, :].sum(axis=0)

            # Update sufficient statistics
            # ss[docs[m],:] dim(Nm,T), self.phi[m,:,:] dim(Nm,T)
            # ss[docs[m], :] += self.phi[m, :Nm, :]


            # # Update eta
            # idx = self.docs[m]
            # # t1 dim(T,Nm)
            # t1 = np.log(beta_s[:, idx]) + c_log_pi - np.log(beta_r[:, idx]) - c_log1_pi
            # # t2 dim(Nm, T)
            # t2 = self.phi[m, :Nm, :]
            # # t3 dim(T, Nm)
            # t3 = t1 * t2.T
            # # t4 dim(Nm,)
            # t4 = t3.sum(axis=0)
            # # dim(Nm,)
            # self.eta[m, :Nm] = t4 / (1 + np.exp(-t4))

             # Update eta
            idx = self.docs[m].astype(int)
             # t1 dim(T,Nm)
            t1 = np.log(beta_s[:, idx]+e) + c_log_pi - np.log(beta_r[:, idx]+e) - c_log1_pi
             # t2 dim(Nm, T)
            t2 = self.phi[m, :Nm, :]
             # t3 dim(T, Nm)
            t3 = t1 * t2.T
             # t4 dim(Nm,)
            t4 = t3.sum(axis=0)
             # dim(Nm,)
            self.eta[m, :Nm] = sigmoid(t4)

        # optimize gamma
        # gamma dim(M,T)  alpha dim(T,) phi.sum(axis=1) dim(M,T)
        self.gamma = self.alpha + self.phi.sum(axis=1)



    def M_step_s(self, beta_s):
        """
        maximize the lower bound of the likelihood.
        This is the M-step of variational EM algorithm for (smoothed) LDA.

        update of alpha follows from appendix A.2 of Blei et al., 2003.
        """
        # update alpha
        self.alpha = self._update(self.alpha, self.gamma, self.M)

        # update beta   
        v_m = self.eta.shape[0]
        v_n = self.eta.shape[1]
        
        # phi dim(M,N,T) eta dim(M,N)
        
        # Update beta_s    
        phi_s = self.phi * self.eta.reshape(v_m, v_n, 1)
        for j in range(self.V):
            #beta
            # beta[:,j] dim(T,),
            t0 = np.array([self._phi_dot_w(phi_s, m, j) for m in range(self.M)]).sum(axis=0)
            beta_s[:, j] = t0        
        beta_s /= beta_s.sum(axis=1).reshape(-1, 1)

        #Update beta 
        phi_r = self.phi * (1 - self.eta.reshape(v_m, v_n, 1))
        for j in range(self.V):
            #beta
            # beta[:,j] dim(T,),
            t0 = np.array([self._phi_dot_w(phi_r, m, j) for m in range(self.M)]).sum(axis=0)
            self.beta[:, j] = t0

        self.beta /= self.beta.sum(axis=1).reshape(-1, 1)

   

    def M_step(self):
        """
        maximize the lower bound of the likelihood.
        This is the M-step of variational EM algorithm for (smoothed) LDA.

        update of alpha follows from appendix A.2 of Blei et al., 2003.
        """
        # update alpha
        self.alpha = self._update(self.alpha, self.gamma, self.M)

        # update beta
        for j in range(self.V):
            #beta
            # beta[:,j] dim(T,),
            self.beta[:, j] = np.array([self._phi_dot_w(self.phi, m, j) for m in range(self.M)]).sum(axis=0)
        self.beta /= self.beta.sum(axis=1).reshape(-1, 1)
    

    def M_step_ss(docs, phi, gamma, alpha, beta, M):
        """
        maximize the lower bound of the likelihood.
        This is the M-step of variational EM algorithm for (smoothed) LDA.

        update of alpha follows from appendix A.2 of Blei et al., 2003.
        """
        # update alpha
        alpha = _update(alpha, gamma, M)

        # update beta
        # beta dim(T,V)  ss dim(V,T)
        beta = ss.T
        beta /= beta.sum(axis=1).reshape(-1, 1)

        return alpha, beta

    def _phi_dot_w(self, _phi, d, j):
        """
        \sum_{n=1}^{N_d} ϕ_{dni} w_{dn}^j
        """
        # doc = np.zeros(docs[m].shape[0] * V, dtype=int)
        # doc[np.arange(0, docs[m].shape[0] * V, V) + docs[m]] = 1
        # doc = doc.reshape(-1, V)
        # lam += phi[m, :Nm, :].T @ doc
        return (self.docs[d] == j) @ _phi[d, :self.N[d], :]

    def _update(self, var, vi_var, const, max_iter=10000, tol=1e-6):
        """
        From appendix A.2 of Blei et al., 2003.
        For hessian with shape `H = diag(h) + 1z1'`

        To update alpha, input var=alpha and vi_var=gamma, const=M.
        To update eta, input var=eta and vi_var=lambda, const=k.
        """
        for i in range(max_iter):
            # store old value
            var0 = var.copy()

            # g: gradient
            psi_sum = psi(vi_var.sum(axis=1)).reshape(-1, 1)
            g = const * (psi(var.sum()) - psi(var)) \
                + (psi(vi_var) - psi_sum).sum(axis=0)

            # H = diag(h) + 1z1'
            z = const * polygamma(1, var.sum())  # z: Hessian constant component
            h = -const * polygamma(1, var)  # h: Hessian diagonal component
            c = (g / h).sum() / (1. / z + (1. / h).sum())

            # update var
            var -= (g - c) / h

            # check convergence
            err = np.sqrt(np.mean((var - var0) ** 2))
            crit = err < tol
            if crit:
                break
        else:
            warnings.warn(f"max_iter={max_iter} reached: values might not be optimal.")

        # print(err)
        print(f"Alpha update took {i} iteration(s)")
        return var



    def vlb(self):
        """
        Average variational lower bound for joint log likelihood.
        """
        a = 1e-10
        lb = 0
        for d in range(self.M):
            lb += (
                gammaln(np.sum(self.alpha))
                - np.sum(gammaln(self.alpha))
                + np.sum([(self.alpha[i] - 1) * dg(self.gamma, d, i) for i in range(self.K)])
            )

            lb -= (
                gammaln(np.sum(self.gamma[d, :]))
                - np.sum(gammaln(self.gamma[d, :]))
                + np.sum([(self.gamma[d, i] - 1) * dg(self.gamma, d, i) for i in range(self.K)])
            )

            for n in range(self.N[d]):
                w_n = int(self.docs[d][n])

                lb += np.sum([self.phi[d][n, i] * dg(self.gamma, d, i) for i in range(self.K)])
                # if(np.any(self.beta[i, w_n] < 0)):
                #     print(self.beta[i, w_n])
                #     if np.any(self.eta < 0):
                #         print('ETA', self.eta)
                #     #input('Press any key to continue')

                lb += np.sum([self.phi[d][n, i] * np.log(self.beta[i, w_n] ) for i in range(self.K)])
                lb -= np.sum([self.phi[d][n, i] * np.log(self.phi[d][n, i] ) for i in range(self.K)])

        return lb / self.M


#my_vocab.index

def insert_seeds(seeds, beta_s, word2id):
    print('Insert seeds')
    for k in range(len(seeds)):
        n = len(seeds[k])
        p = 1.0 / n
        for i in range(n):
            try:
                print('word2id index')
                idx = word2id.index(seeds[k][i])
                print(f"({k},{idx}) {p:.3f}")
                beta_s[idx, k] = np.log(p)
            except Exception as e:
                print(f'ERROR {e}')
            
            

def print_beta(beta, n_words, id2word):
        # Assume beta dim(V,T)
        print(f"beta shape:{beta.shape}")
        t2 = beta
        #t2 = np.transpose(beta)
        K = t2.shape[0]
        N = t2.shape[1]
        for k in range(K):
            print('\nTopic #{}'.format(k))
            i = 0
            a = np.argsort(-t2[k,:])
            for id in a:
                print(f'{id2word[id]} {t2[k,id]:.3f}')                
                i += 1
                if i > n_words:
                        break
                        



print("========== TRAINING STARTED ==========")

data = pickle.load( open( "test2.pkl", "rb" ) )


num_topics = 5

print(f"Total numbers of papers:{len(data)}")
print(f"-------Numbers of Topics: {num_topics}--------\n")

text_data = data[:]

print('Generate corpus and dictionary!')
t1 = time.time() 
my_docs = []
my_vocab = []

for text in text_data:
    for word in text:
        if not word in my_vocab:
            #print(f'{word}')
            my_vocab.append(word)

#print(f'length of my_vocab:{len(my_vocab)}\n{len(my_vocab[0])}')
#print(my_vocab)
#print(my_vocab.index('tick'))

#sys.exit()

total_num_words = 0

for text in text_data:
    x = []
    for word in text:
        i = my_vocab.index(word)
        x.append(i)
        total_num_words += 1
    my_docs.append(np.array(x))
t2 = time.time()
print(f'Generate corpus took {(t2-t1):.3f}')
print(f'Total numbers of words in dataset:{total_num_words}')

            
#print(f'Vocab:{my_vocab[:10]}')
print(f'Vocab size:{len(my_vocab)}')
print(f'Length of {len(my_docs)}')
#print(f'Doc:{my_docs[:10]}')    

# generate beta_s
len_vocab = len(my_vocab)

seed_topic_list = [
['loan', 'bond', 'fund', 'model', 'asset']
]

n_topic = num_topics

beta_sum = np.zeros((n_topic, len_vocab))
beta_s = np.ones((n_topic, len_vocab)) * 1e-8
pi = 0.5
c_log_pi = np.log(pi)
c_log1_pi = np.log(1 - pi)




'''
for k in range(n_topic):
    n = len(s_list[k])
    #p = 1.0 / n
    p = 0.9
    for i in range(n):
        idx = my_vocab.index(s_list[k][i])
        #print('({},{}) {:.3f}'.format(k,idx,p))
        beta_s[k, idx] = p
'''

#insert_seeds(seed_topic_list, beta_s, my_vocab)
seeds = seed_topic_list

for k in range(len(seeds)):
    n = len(seeds[k])
    p = 1.0 / n
    for i in range(n):
        try:
            print('word2id index')
            idx = my_vocab.index(seeds[k][i])
            print(idx)
            #print(f"({k},{idx}) {p:.3f}")
            beta_s[k, idx] = p 
        except Exception as e:
            print(f'ERROR {e}')

            
#print_beta(beta_s, 5, my_vocab) 


np.random.seed(0)

lda = GuidedLda(my_docs, my_vocab, n_topic)
print(f'{lda.n_sum=}')
print(f'{lda.n_max=}')


#sys.exit(0)

#print_topic(beta_s, 5, my_vocab)
#print_topic(beta, 5, my_vocab)

N_EPOCH = 4
TOL = 1e-2

verbose = True
lb = -np.inf


for epoch in range(N_EPOCH):
    # store old value

    lb_old = lb
    t1 = time.time()

    #ss = np.zeros((V, k))

    # Variational EM
    # lda.E_step()
    lda.E_step_s(lda.beta, beta_s)
    
    t2 = time.time()

    #lda.M_step()
    lda.M_step_s(beta_s)
    
    t3 = time.time()
    print(f"Iter:{epoch:04} Time: E:{t2-t1:.3f} M:{t3-t2:.3f} Total:{t3-t1:.3f}")
    # check anomaly
    if np.any(np.isnan(lda.alpha)):
        print("NaN detected: alpha")
        break

    # check convergence
    lb = lda.vlb()
    #lb = 10
    err = abs(lb - lb_old)

    # check anomaly
    if np.isnan(lb):
        print("NaN detected: lb")
        break

    if verbose:
        print(f"{epoch: 04}:  variational_lb: {lb: .3f},  error: {err: .3f}")

    if err < TOL:
        break
else:
    warnings.warn(f"max_iter reached: values might not be optimal.")

print(f"Alpha after training:{lda.alpha}")
print(" ========== TRAINING FINISHED ==========\n")

ratio = np.sum(lda.eta)/lda.n_sum

beta_sum = 0.5 * lda.beta + 0.5 * beta_s

for i in range(lda.K):
    print(f"TOPIC {i:02}: {n_most_important(lda.beta[i], 9)}")

print('\n------------Regular topic-words-----------------') 
print_beta(lda.beta, 5, my_vocab) 

print('\n------------Seeded topic-words-----------------') 
print_beta(beta_s, 5, my_vocab)


print('\n------------Combined topic-words-----------------') 
print_beta(beta_sum, 5, my_vocab)

print(f'{lda.eta.size=}')
print(f'ETA SUM:{np.sum(lda.eta)/lda.n_sum}')

    
'''
n_sample = 10000
theta_hat = np.array([np.random.dirichlet(lda.gamma[d], n_sample).mean(0) for d in range(lda.M)])
theta_hat /= theta_hat.sum(1).reshape(-1, 1)

plt.figure(figsize=(8,8))
plt.subplot(121)
n_plot_words = 150
sns.heatmap(lda.beta.T[:n_plot_words], xticklabels=[], yticklabels=[])
#sns.heatmap(beta_s.T[:n_plot_words], xticklabels=[], yticklabels=[])
plt.xlabel("Topics", fontsize=14)
plt.ylabel(f"Words[:{n_plot_words}]", fontsize=14)
plt.title("topic-word distribution", fontsize=16)

plt.subplot(122)
sns.heatmap(theta_hat, xticklabels=[], yticklabels=[])
plt.xlabel("Topics", fontsize=14)
plt.ylabel("Documents", fontsize=14)
plt.title("document-topic distribution", fontsize=16)

plt.tight_layout();
plt.show()

#print(lda.eta)

#print_topic(beta, 5, my_vocab)
#print_topic(beta_s, 5, my_vocab)
if(np.any(lda.eta <= 0)):
    print("ETA <=0!!!!")
print(f'ETA SUM:{np.sum(lda.eta)/lda.eta.size}')
'''


