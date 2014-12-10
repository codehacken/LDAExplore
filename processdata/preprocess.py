__author__ = 'Ashwin'
__email__ = 'gashwin1@umbc.edu'

"""
Basic Pre-processing Operations.
"""

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import operator

stemmer = PorterStemmer()


# Performing the Stemming operation.
def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed


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
    feature_names = get_feature_names(tfidf)

    for doc in tfs:
        term_list = []
        for term in doc.nonzero()[1]:
            term_list.append(feature_names[term])

        # Sort the dictionary values and then append to the list.
        # term_list.append(sorted(term_dict.items(), key=operator.itemgetter(1)))
        s_words.append(term_list)

    return s_words
