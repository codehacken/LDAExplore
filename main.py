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

from processdata.fileops import FileReader
from processdata.lda import LDAVisualModel
from processdata.fileops import write_rank_to_file

num_of_topics = 20
num_of_passes = 50
num_of_words = 10

# Location of the data.
data_loc = '20_newsgroups/alt.temp/9976'
data_dir_loc = '20_newsgroups/alt.temp/'
data_sects = 'server/data/info_vis_abstracts.csv'
title_list = 'server/data/key_vis_title.csv'

final_data_file = 'server/data/data.csv'

if __name__ == "__main__":

    # Read the directory
    reader = FileReader()
    #reader.read_file(data_loc)
    #reader.read_dir(data_dir_loc)
    reader.read_text_sections(data_sects)

    # Get the token list
    word_corpus = reader.get_token_list()

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

    # Print the topic information to a file.
    write_rank_to_file(doc_to_word, doc_top_rank, num_of_words, num_of_topics, title_list, final_data_file)
