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
from processdata.fileops import write_top_word_to_file
from processdata.fileops import write_top_word_matrix_to_file

num_of_words = 20   # Totally 5792 in our dataset
num_of_topics = 20
para_alpha = 1 / num_of_topics  #1 / num_of_topics  # 'symmetric'(de),'asymmetric','auto'
num_of_passes = 1  # Default is 1, 10 --> much worse
num_of_updates = 1  # Default is 1 in LDA, 50 give too many repeated words, and one-document words after re-ranking
num_of_trunks = 100   # Default 2000, 200 is good. Large-->too many repeated words, and one-document words after re-ranking; Small->less repeated but words are too general meaning
num_of_iterations = 10  # Default is 50
num_of_eval = 10    # Default is 10
para_gamma_t = 0.001  # Default is 0.001
para_kappa = 0.5  # Default is 0.5
para_tao = 1.0  # Default is 1.0
para_eta = None  # Default is None
min_df_cutoff = 0.3  # 0.3

# Location of the data.
data_loc = '20_newsgroups/alt.temp/9976'
#data_dir_loc = '../20news/20_newsgroups/alt.atheism' change line 79/80 and 90/91
data_sects = 'server/data/TVCG_abstract.csv'
title_list = 'server/data/TVCG_title.csv'

# List of files that are written to.
prob_data_file = 'server/data/TVCG_prob.csv'
rank_data_file = 'server/data/TVCG_rank.csv'
top_hier_data_file = 'server/data/top_hier_data.json'
top_word_data_file = 'server/data/TVCG_top_words.txt'
re_top_word_data_file = 'server/data/TVCG_re_rank_top_words.txt'
top_word_matrix_file = 'server/data/TVCG_top_words_matrix.csv'
#s_words = ['data', 'visualization', 'visual', 'approach', 'analysis', 'study', 'techniques',
#           'interactive', 'results', 'design', 'paper', 'user',  'information', 'based', 'system',
#           'present', 'time', 'different', 'use', 'using', 'used']

if __name__ == "__main__":

    # Read the directory
    tfidf_tokenizer = FileReader()
    tfidf_tokenizer.read_file_text(data_sects)
#    tfidf_tokenizer.read_data_dir_loc(data_dir_loc)
    # Get the token list
    raw_text = tfidf_tokenizer.get_token_list()
    [tfidf, tfs] = preprocess.vectorize(raw_text, min_df_cutoff)  ###
    s_words = preprocess.build_stop_word_list(tfidf, tfs)  ###
    e_words = preprocess.get_stop_words(tfidf)  ###

    # Get the set of words.
    # Read the file again for LDA tokens.
    lda_tokenizer = FileReader()
    lda_tokenizer.read_text_sections(data_sects, s_words, e_words)
#    lda_tokenizer.read_text_sections_dir(data_dir_loc, s_words, e_words)
    word_corpus = lda_tokenizer.get_token_list()  # Generate lda_tokenizer.token_list with repeated words

    # word_corpus = preprocess.singularize(word_corpus)
    [word_corpus, stem_relation] = preprocess.stem(word_corpus)
    stem_map = preprocess.gen_stem_map(stem_relation)  # dictionary that map back

    # Perform LDA.
    lda = LDAVisualModel(word_corpus)  # token2id
    lda.create_word_corpus(word_corpus)  # create mm: 8 * list of (token_id, token_count)

    # Train the LDA model for specific number of topics
    # and iterations.
    lda.train_lda(num_of_topics, para_alpha, num_of_passes, num_of_updates,
                  num_of_trunks, num_of_iterations, num_of_eval,
                  para_gamma_t, para_kappa, para_eta)  # id2token, lda.lda
        # Get lda.lda.expElogbeta: (matrix) num_topic * num_corpus_word
        # Get lda.id2word.id2token because we feed in lda.id2word when training
        #TODO: para_tao has been removed from the 2nd last position temporarily. Add it back later.

#   C-LDA
#    must_link = tfidf_tokenizer.read_must_link(must_link_file)
#    cannot_link = tfidf_tokenizer.read_cannot_link(cannot_link_file)
#    lda.train_clda(num_of_topics, must_link, cannot_link, num_of_passes, num_of_updates,
#                  num_of_trunks, num_of_iterations, num_of_eval,
#                  para_gamma_t, para_kappa, para_tao, para_eta)

    lda.id2word.id2token = preprocess.inverse_stem(lda.id2word.id2token, stem_map)
        # Only clean lda.id2word.id2token, NOT cleaning anything else
        # Not needed to clean lda.id2word.token2id, lda.lda.id2word

    # Generate matrices  # OK
    matrix_doc_top = lda.generate_doc_topic()
    doc_top_rank = lda.generate_doc_topic_rank()
        # *******
        # Use: for idx, doc in enumerate(self.lda[self.mm])
        # To generate num_doc lists. Each list has many pairs of
        # (top_idx, top_prob). Ex: (2, 0.0182), (6, 0.0155)

    matrix_top_word = lda.lda.expElogbeta
    matrix_doc_word = lda.generate_doc_word(matrix_doc_top)
    [top_word, top_word_index] = lda.generate_top_words(num_of_words)

    # Get the topic corpus.
    topics = lda.get_lda_corpus(num_of_topics, num_of_words)  #
        # Print topic_words (Decide how many) for each topic
        # Each has num_of_words(20) pairs of ('0.023', 'network')

    # Isolate top words for documents.
    doc_to_word = lda.gen_doc_top_words(topics, matrix_doc_top)  #
        # Print doc_words for each document
        # Each has num_or_words(20) pairs of ('0.023', 'network')

    # Generate the topic hierarchy.  #OK
    top_hier = lda.gen_topic_hierarchy(topics)  #
    top_word_o = lda.gen_topic_words(topics)  #

    # Evaluation
    doc_word_index_set = lda.doc_word_index_set()
    top_coherence = lda.topic_coherence(top_word_index, num_of_words, doc_word_index_set)  # lda.mm is in self
        # Need the indices of the first 20 words, and then go to
        # those columns in doc_word_times matrix
    coherence = sum(top_coherence) / num_of_topics
    perplexity = lda.perplexity(matrix_doc_word, doc_word_index_set)

    # Print the topic information to a file.  #OK
    write_prob_to_file(doc_to_word, matrix_doc_top, num_of_words, num_of_topics, title_list, prob_data_file)
    write_rank_to_file(doc_to_word, doc_top_rank, num_of_words, num_of_topics, title_list, rank_data_file)
    write_top_hier_to_file(top_hier, top_hier_data_file)
    write_top_word_to_file(top_word_o, top_word_data_file, top_coherence, coherence, perplexity)
    write_top_word_matrix_to_file(lda.id2word.id2token, matrix_top_word, top_word_matrix_file)

    # Re-rank topic keywords
    re_rank_top_word = lda.top_word_re_rank(num_of_words, 2)
        # CHANGED FOR DEBUGGING  num_of_words)
        # 1:SUM 2:LOG

    # Print the re-ranked topic keywords
    write_top_word_to_file(re_rank_top_word, re_top_word_data_file, top_coherence, coherence, perplexity)
