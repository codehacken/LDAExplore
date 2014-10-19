__author__ = 'ashwin'
"""
Basic LDA module that is used in the project.
"""

from gensim import corpora, models, similarities
#from itertools import chain

class LDAVisualModel:
    def __init__(self, word_corpus):
        """
        The LDAVisualModel requires list of word lists from the
        document corpus. Each list of words represents a document.
        :param word_corpus: [[<words>],[],[]]
        """
        self.id2word = corpora.Dictionary(word_corpus)
        self.mm = []
        self.lda = None

    def create_word_corpus(self, word_corpus, store_corpus=False, store_loc='dicts/corpus.mm'):
        """
        :param word_corpus: word_corpus: [[<words>],[],[]]
        :param store_corpus: boolean to store the serialized corpus or not.
        :param store_loc: Defines the location where the file is to be stored.
        """
        for text in word_corpus:
            self.mm.append(self.id2word.doc2bow(text))

        if store_corpus:
            corpora.MmCorpus.serialize(store_loc, corpus)

    def train_lda(self, num_top=2, update_t=1, chunks=10000, num_pass=1):
        """
        :param num_top: The number of topics for which LDA trains.
        :param update_t:
        :param chunks:
        :param num_pass: The number of passes that LDA executes on the data.
        """
        self.lda = models.LdaModel(corpus=self.mm, id2word=self.id2word, num_topics=num_top,
                                       update_every=update_t, chunksize=chunks, passes=num_pass)

    def get_lda_corpus(self):
        """
        Get the topic associated with each document.
        """
        topics = []
        if self.lda:
            for topic in self.lda.print_topics():
                topics.append(topic)

        return topics

