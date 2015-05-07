__author__ = 'Ashwin'
__email__ = 'gashwin1@umbc.edu'

"""
Basic Pre-processing Operations.
"""

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import operator

import inflect
p = inflect.engine()

stemmer = PorterStemmer()


# Performing the Stemming operation.
def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed

def singularize(word_list):
    word_list_singular = []
    for document in word_list:
        document_singular = []
        for word in document:
            w_plural = p.singular_noun(word) # if singular -> false, if plural -> singular form
            if w_plural is not False:
                document_singular.append(w_plural)
            else:
                document_singular.append(word)
        word_list_singular.append(document_singular)
    return word_list_singular

def stem(word_list):
    word_list_singular = []
    stem_relation = {} # Empty dictionary
    for document in word_list:
        document_singular = []
        for word in document:
            word_singular = stemmer.stem(word)
            document_singular.append(word_singular)
            if word_singular not in stem_relation:
               stem_relation[word_singular] = set()
            stem_relation[word_singular].add(word)

        word_list_singular.append(document_singular)
    return [word_list_singular, stem_relation]

def gen_stem_map(stem_relation):
    stem_map = {} # Empty dictionary
    for item in stem_relation:  # item - items, itemize
        word_shortest = ""
        for word in stem_relation[item]:
            if len(word) < len(word_shortest) or len(word_shortest) == 0:
                word_shortest = word
        stem_map[item] = word_shortest

    return stem_map

#def inverse_stem(topics, stem_map):
#    topics_m = topics
#    for i, topic in enumerate(topics_m):
#        for j, word in enumerate(topic):
        #   topics_m[i][j][1] = stem_map[word[1]]  # tuple is immutable
#            topics_m[i][j] = word[0], stem_map[word[1]]
#    return topics_m

def inverse_stem(id2token, stem_map):
    id2token_m = id2token
    for i, token_key in enumerate(id2token_m):
            id2token_m[i] = stem_map[id2token_m[i]]
    return id2token_m

# tokenize each document.
def tokenize(text):
    tokens = word_tokenize(text)
    #stems = stem_tokens(tokens)
    #return stems
    return tokens


# Return the tfidf and tfs values for each document.
def vectorize(text_list, min_df):
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfidf.min_df = min_df
    tfs = tfidf.fit_transform(text_list)
    return [tfidf, tfs]


# Get the stop words.
def get_stop_words(tfidf):
    stop_words = []
    for word in tfidf.get_stop_words():
        stop_words.append(word)

    return stop_words


# Get the stop words.
def get_feature_names(tfidf):
    return tfidf.get_feature_names()


# Get the words for each document.
# Build a stop word list for each document.
def build_stop_word_list(tfidf, tfs):
    s_words = []
    feature_names = get_feature_names(tfidf)   # stop word list, like based, data, new, paper, ...

    for doc in tfs: # doc = (0,0) 0.118; (0,14) 0.188; (0,3) 0.473; doc.nonzero()[1] = 0...14...3
        term_list = []
        for term in doc.nonzero()[1]:
            term_list.append(feature_names[term])  # doc.nonzero()[1] is used as index of feature_names

        # Sort the dictionary values and then append to the list.
        # term_list.append(sorted(term_dict.items(), key=operator.itemgetter(1)))
        s_words.append(term_list)

    return s_words
