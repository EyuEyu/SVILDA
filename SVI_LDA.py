# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:31:46 2019

@author: default
"""
import json
from nltk.tokenize import wordpunct_tokenize
import numpy as np
from scipy.special import psi #digamma function
import heapq
import random
f=open("arxiv_math_cs.json","r")
content=json.load(f)
f.close()
f=open("vocab.json","r")
vocab=json.load(f)
f.close()

np.random.seed(2019)
def getAllDocs():
    docs=[]
    for i in range(len(content)):
        docs.append(content[i]["abstract"].lower())
    return docs
def getVocab():
    return vocab

def parseDocs(doc, vocab):
    docWords=wordpunct_tokenize(doc)
    docWords_dict={}
    for word in docWords:
        if word in vocab:
            w_index=vocab[word]
            if w_index in docWords_dict.keys():
                docWords_dict[w_index]+=1
            else:
                docWords_dict[w_index]=1
    return list(docWords_dict.keys()),list(docWords_dict.values())

######func for LDA

def dirichlet_log_expectation(alpha):
    if len(np.shape(alpha))==1:
        return (psi(alpha)-(np.sum(alpha)))
    return (psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

def beta_log_expectation(a,b,k):
    Elog_a = psi(a)-psi(a+b)
    Elog_b = psi(b)-psi(a+b)
    Elog_beta=np.zeros(k)
    Elog_beta[0]=Elog_a[0]
    for i in range(1,k):
        Elog_beta[i]=Elog_a[i] + np.sum(Elog_b[0:i])
    return Elog_beta

######SVI_LDA
class SVI_LDA():
    def __init__(self,docs, vocab, K, alpha=0.5, eta=0.02, tau=1000, kappa=0.7, iterations=1726):
        self.vocab = vocab
        self.docs = docs
        self.K = K
        self.D=len(docs)
        self.V=len(vocab)
        self.alpha=alpha
        self.eta=eta
        self.tau=tau
        self.kappa=kappa
        self.iterations=iterations
        self.Lambda=np.random.gamma(100,1/100,size=(self.K,self.V))
        self.expElogBeta=np.exp(dirichlet_log_expectation(self.Lambda))
        self.count=0
        self.iterations=iterations
        self.store_topic_probability=[]
        self.change_threshold=0.001
        
        for i in range(K):
            self.store_topic_probability.append([1/self.K])
            
    def update_local(self,doc):
        words,counts=doc
        newdoc=[]
        N_d=sum(counts)
        phi_d=np.zeros((self.K,N_d))
        gamma_d=np.random.gamma(100,1/100,(self.K))
        expElogTheta_d=np.exp(dirichlet_log_expectation(gamma_d))
        for i in range(len(counts)):
            for j in range(counts[i]):
                newdoc.append(words[i])
        while True:
            for i in range(len(newdoc)):
                phi_d[:,i] =np.multiply(expElogTheta_d,self.expElogBeta[:,newdoc[i]])
                phi_d[:,i] = phi_d[:,i]/np.sum(phi_d[:,i])
               
            gamma_new=self.alpha + np.sum(phi_d,axis=1)
            change=np.mean(abs(gamma_new-gamma_d))
            if change<self.change_threshold:
                break
            
            gamma_d=gamma_new
            expElogTheta_d=np.exp(dirichlet_log_expectation(gamma_d))
        newdoc=np.array(newdoc)
        return phi_d, newdoc, gamma_d
    
    def update_Global(self,phi_d,doc):
        Lambda_d=np.zeros((self.K,self.V))
        for k in range(self.K):
            phi_dk=np.zeros(self.V)
            for i in range(len(doc)):
                phi_dk[doc[i]]+=phi_d[k][i]
            Lambda_d[k] = self.eta + self.D*phi_dk
        rho =(self.count+self.tau)**(-self.kappa)
        self.Lambda=(1-rho)*self.Lambda+rho*Lambda_d
        self.expElogBeta=np.exp(dirichlet_log_expectation(self.Lambda))

    def Probability(self):
        topic_probability=np.sum(self.Lambda,axis = 1)
        word_probability=np.sum(self.Lambda,axis = 0)
        return topic_probability/np.sum(topic_probability),word_probability/np.sum(word_probability)
    
    def get_Topics(self,docs=None):
        topic_probability,word_probability=self.Probability()
        if docs==None:
            docs=self.docs
        results=np.zeros((len(docs),self.K))
        for i,doc in enumerate(docs):
            words,counts = parseDocs(doc,self.vocab)
            for j in range(self.K):
                media=[self.Lambda[j][word]/word_probability[word] for word in words]
                doc_probability=[np.log(media[m])*counts[m] for m in range(len(media))]
                results[i][j]=sum(doc_probability) +np.log(topic_probability[j])
        finalResults =np.zeros(len(docs))
        for d in range(len(docs)):
            finalResults[d] =np.argmax(results[d])
        return finalResults,topic_probability
    
    def get_Topic_words(self,N):
        for i in range(len(self.Lambda)):
            Lambda_i=list(self.Lambda[i])
            words_index = list(map(Lambda_i.index, heapq.nlargest(N, Lambda_i)))
            print("\nTopic "+str(i)+":")
            for j in range(N):
                print(list(self.vocab.keys())[list(self.vocab.values()).index(words_index[j])])
    def Run(self):
        for i in range(self.iterations):
            r=random.randint(0,self.D-1)
            print(i)
            doc=parseDocs(self.docs[r],self.vocab)
            phi_d, newdoc, gamma_d =self.update_local(doc)
            self.update_Global(phi_d,newdoc)
            self.count+=1
if __name__ == '__main__':
    docs = getAllDocs()
    vocab= getVocab()
    K=2
    model=SVI_LDA(docs=docs,vocab=vocab,K=K)
    model.Run()
    results,prob=model.get_Topics()
    model.get_Topic_words(10)
    
    
    subject_list=[]
    for i in range(len(content)):
        if content[i]["subject"]=="math":
            subject_list.append(1)
        else:
            subject_list.append(0)
    right=0
    for i in range(len(subject_list)):
        if subject_list[i]==results[i]:
            right+=1
    
    print("######accuracy:",max(1-right/len(subject_list),right/len(subject_list)))
'''
math topic:  
problem
paper
space
number
function
algorithm
time
study
prove
set

cs topic:
data
based
model
learning
method
network
paper
models
proposed
performance

Accuracy for classification: 0.8267670915411356 #unstable, not know why.

'''
                
                
                
        
        
        
        
    