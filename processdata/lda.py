__author__ = 'Ashwin'
__email__ = 'gashwin1@umbc.edu'

"""
Basic LDA module that is used in the project.
"""
import re
from gensim import corpora, models
import operator


class LDAVisualModel:
    def __init__(self, word_corpus, scaling_factor=100000):
        """
        The LDAVisualModel requires list of word lists from the
        document corpus. Each list of words represents a document.
        :param word_corpus: [[<words>],[],[]]
        """
        self.id2word = corpora.Dictionary(word_corpus)
        self.mm = []
        self.lda = None
        self.scaling_factor = scaling_factor

    def create_word_corpus(self, word_corpus, store_corpus=False, store_loc='dicts/corpus.mm'):
        """
        :param word_corpus: word_corpus: [[<words>],[],[]]
        :param store_corpus: boolean to store the serialized corpus or not.
        :param store_loc: Defines the location where the file is to be stored.
        """
        for text in word_corpus:
            self.mm.append(self.id2word.doc2bow(text))
	
        if store_corpus:
            corpora.MmCorpus.serialize(store_loc, word_corpus)

    def train_lda(self, num_top=2, update_t=1, chunks=10000, num_pass=1):
        """
        :param num_top: The number of topics for which LDA trains.
        :param update_t:
        :param chunks:
        :param num_pass: The number of passes that LDA executes on the data.
        """
        self.lda = models.LdaModel(corpus=self.mm, id2word=self.id2word, num_topics=num_top,
                                   update_every=update_t, chunksize=chunks, passes=num_pass)

    def get_lda_corpus(self, num_of_topics=10, num_of_words=10):
        """
        Get the topic associated with each document.
        """
        topics = []
        if self.lda:
            for topic in self.lda.print_topics(num_of_topics, num_of_words):
                regex = re.findall(r'(0\.[0-9]*)\*([0-9a-z]*)', topic, re.M | re.I)
                topics.append(regex)

        return topics

    def generate_doc_topic(self):
        # Find the number of topics.
        num_topics = self.lda.num_topics

        # Build the topic - document matrix.
        doc_top = []
        for idx, doc in enumerate(self.lda[self.mm]):
            doc_top.append([0] * num_topics)
            for topic in doc:
                doc_top[idx][topic[0]] = topic[1]

        return doc_top

    def generate_doc_topic_rank(self):
        # Find the number of topics.
        num_topics = self.lda.num_topics

        doc_top_rank = []

        # Build the topic - document matrix.
        for idx, doc in enumerate(self.lda[self.mm]):
            top_prob = [0] * num_topics
            top_rank = [0] * num_topics

            # This constructs the topic probability list.
            for topic in doc:
                top_prob[topic[0]] = topic[1]

            # Construct the ranks.
            prob_rank = sorted(top_prob, reverse=True)
            top_sort = sorted(range(len(top_prob)), key=lambda k: top_prob[k],
                              reverse=True)

            # Create a new list with the ranks.
            for rank, topic in enumerate(top_sort):
                if prob_rank[rank] > 0:
                    top_rank[topic] = rank
                else:
                    top_rank[topic] = num_topics - 1

            doc_top_rank.append(top_rank)

        return doc_top_rank

    @staticmethod
    def gen_topic_hierarchy(topics):
        # Create the hierarchy.
        top_word_hier = {"children": []}

        for idx, word_list in enumerate(topics):
            children = []

            # Create the children nodes - words.
            for word in word_list:
                children.append({"name": word[1], "size": word[0],
                                 "value": float(word[0])*1000, "url": "javascript:void(0)"})

            # Add the list to current hierarchy.
            top_word_hier["children"].append({"name": "T" + str(idx), "children": children})

        top_word_hier["name"] = 'Topics'
        return top_word_hier

    @staticmethod
    def gen_doc_top_words(topics, doc_top):
        # This maintains the top words list for each document.
        doc_to_word = []

        # Check the probability of the topic and the word
        # distribution in it.

        for doc in doc_top:
            tmp_word_prob = {}
            for idx, top_prob in enumerate(doc):
                if top_prob > 0:
                    for word in topics[idx]:
                        if word not in tmp_word_prob:
                            tmp_word_prob[word[1]] = float(word[0])*top_prob
                        else:
                            tmp_word_prob[word[1]] += float(word[0])*top_prob

            # Sort the dictionary
            sorted_word_prob = sorted(tmp_word_prob.items(), key=operator.itemgetter(1), reverse=True)
            doc_to_word.append(sorted_word_prob)

        return doc_to_word
