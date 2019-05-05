# SVILDA
Stochastic Variational Inference for Latent Dirichlet Allocation

arxiv_math_cs.json: This is the data.
The data is from arxiv paper abstracts, including math and cs, 1726 papers in total. The number of math paper abstracts is 762. The number of cs paper abstracts is 964.

vocab.json: This is vocab in model.
vocab comes from paper abstracts after the elimination of stop words, 25587 in total.

SVI_LDA.py is the main func.

Result:
math topic words:  
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

cs topic words:  
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

Accuracy for math_cs classification: 0.8267670915411356 #unstable, not know why.



