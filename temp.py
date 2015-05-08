#!/usr/bin/python
from gensim import corpora, models, similarities
from itertools import chain 

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

print "TEXT"
print texts

# remove words that appear only once
all_tokens = sum(texts, [])

print "ALL TOKENS"
print all_tokens

tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

print "TOKEN ONCE"
print tokens_once

texts = [[word for word in text if word not in tokens_once] for text in texts]

print "TEXTS"
print texts

print "NOW LDAing...."

# Create Dictionary.
id2word = corpora.Dictionary(texts)
print "Dictionary"
print id2word

# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in texts]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=3, \
                               update_every=1, chunksize=10000, passes=1)

# Prints the topics.
for top in lda.print_topics():
  print top
print

# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]

# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
scores = list(chain(*[[score for topic,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print threshold
print

cluster1 = [j for i,j in zip(lda_corpus,documents) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,documents) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,documents) if i[2][1] > threshold]

print cluster1
print cluster2
print cluster3

print "TEST -----"

import re
original_string = open('foo.txt').read()
new_string = re.sub('[^a-zA-Z0-9\n\.]', ' ', original_string)
open('bar.txt', 'w').write(new_string)

