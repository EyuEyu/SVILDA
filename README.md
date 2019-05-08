# SVILDA
Stochastic Variational Inference for Latent Dirichlet Allocation

arxiv_math_cs.json: This is the data.
The data is from arxiv paper abstracts, including math and cs, 1726 papers in total. The number of math paper abstracts is 762. The number of cs paper abstracts is 964.

vocab.json: This is vocab in model.
vocab comes from paper abstracts after the elimination of stop words, 25587 in total.

SVI_LDA.py is the main func.

Result:

math topic Top 10 words:  
1. problem 
2. paper 
3. space 
4. number 
5. function 
6. algorithm 
7. time 
8. study 
9. prove 
10. set 

cs topic Top 10 words:  

1. data 
2. based 
3. model 
4. learning 
5. method 
6. network 
7. paper 
8. models 
9. proposed 
10. performance 

Accuracy for math_cs classification: 82.96%±2.1%



