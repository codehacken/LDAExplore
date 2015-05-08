#!/usr/bin/python
__author__ = 'Ashwin'
__email__ = 'gashwin1@umbc.edu'

"""
CMSC 691 - Final Project
Visualizing Topics using LDA

AUTHOR: Ashwinkumar Ganesan.
        Kiante Branley

Latent Dirichlet Allocation is a topic modelling method that calculates
the joint probabilities of words across documents and generates the
two distributions:
1. The distribution of topics for each document.
2. The distribution of words for each topic.

This project tries to visualize the topics, the documents and the clusters
that are generated.
"""

"""
E.g for other file types.

# Read the directory
reader = FileReader()
reader.read_file(data_loc)
reader.read_dir(data_dir_loc)
reader.read_text_sections(data_sects)
"""

# File Operations
from processdata.fileops import FileReader
from processdata import preprocess

# LDA Operations.
from processdata.lda import LDAVisualModel
from processdata.fileops import write_rank_to_file
from processdata.fileops import write_prob_to_file
from processdata.fileops import write_top_hier_to_file

num_of_topics = 10
num_of_passes = 50
num_of_words = 10
min_df_cutoff = 0.3

# Location of the data.
data_loc = '20_newsgroups/alt.temp/9976'
data_dir_loc = '20_newsgroups/alt.temp/'
data_sects = 'server/data/info_vis_abstracts.csv'
title_list = 'server/data/key_vis_title.csv'

# List of files that are written to.
prob_data_file = 'server/data/prob_data.csv'
rank_data_file = 'server/data/rank_data.csv'
top_hier_data_file = 'server/data/top_hier_data.json'

#s_words = ['data', 'visualization', 'visual', 'approach', 'analysis', 'study', 'techniques',
#           'interactive', 'results', 'design', 'paper', 'user',  'information', 'based', 'system',
#           'present', 'time', 'different', 'use', 'using', 'used']

if __name__ == "__main__":

    # Read the directory
    tfidf_tokenizer = FileReader()
    tfidf_tokenizer.read_file_text(data_sects)

    # Get the token list
    raw_text = tfidf_tokenizer.get_token_list()
    [tfidf, tfs] = preprocess.vectorize(raw_text, min_df_cutoff)
    s_words = preprocess.build_stop_word_list(tfidf, tfs)
    e_words = preprocess.get_stop_words(tfidf)

    # Get the set of words.
    # Read the file again for LDA tokens.
    lda_tokenizer = FileReader()
    lda_tokenizer.read_text_sections(data_sects, s_words, e_words)
    word_corpus = lda_tokenizer.get_token_list()

    # Perform LDA.
    lda = LDAVisualModel(word_corpus)
    lda.create_word_corpus(word_corpus)

    # Train the LDA model for specific number of topics
    # and iterations.
    lda.train_lda(num_of_topics, num_of_passes)

    # Generate the document to topic matrix.
    doc_top = lda.generate_doc_topic()
    doc_top_rank = lda.generate_doc_topic_rank()

    # Get the topic corpus.
    topics = lda.get_lda_corpus(num_of_topics, num_of_words)

    # Isolate top words for documents.
    doc_to_word = lda.gen_doc_top_words(topics, doc_top)

    # Generate the topic hierarchy.
    top_hier = lda.gen_topic_hierarchy(topics)

    # Print the topic information to a file.
    write_prob_to_file(doc_to_word, doc_top, num_of_words, num_of_topics, title_list, prob_data_file)
    write_rank_to_file(doc_to_word, doc_top_rank, num_of_words, num_of_topics, title_list, rank_data_file)
    write_top_hier_to_file(top_hier, top_hier_data_file)

