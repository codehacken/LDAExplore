__author__ = 'Ashwin'
__email__ = 'gashwin1@umbc.edu'

"""
Basic LDA module that is used in the project.
"""
import re
from gensim import corpora, models
import operator
import numpy  # for top_word re-rank
import scipy  # for gmean

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

    def create_word_corpus(self, word_corpus, store_corpus=False, store_loc='server/data/TVCG_corpus.mm'):
        """
        :param word_corpus: word_corpus: [[<words>],[],[]]
        :param store_corpus: boolean to store the serialized corpus or not.
        :param store_loc: Defines the location where the file is to be stored.
        """
        for text in word_corpus:
            self.mm.append(self.id2word.doc2bow(text))
	
        if store_corpus:
            corpora.MmCorpus.serialize(store_loc, word_corpus)

    def train_lda(self, num_top=10, alpha1='symmetric', num_pass=1, update_t=1, chunks=2000, num_iter=50,
                  num_eval=10, gamma_t=0.001, kappa=0.5, tao=1.0, eta=None):
        """
        :param num_top: The number of topics for which LDA trains.
        :param update_t:
        :param chunks:
        :param num_pass: The number of passes that LDA executes on the data.
        """
        self.lda = models.LdaModel(corpus=self.mm, id2word=self.id2word,
                                   num_topics=num_top, alpha=alpha1,
                                   passes=num_pass, update_every=update_t,
                                   chunksize=chunks, iterations=num_iter,
                                   eval_every=num_eval, gamma_threshold=gamma_t,
                                   decay=kappa, offset=tao, eta=eta)
        #(corpus=None, num_topics=100, id2word=None, distributed=False,
        # chunksize=2000, passes=1, update_every=1, alpha='symmetric',
        # eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50,
        # gamma_threshold=0.001)

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
        for idx, doc in enumerate(self.lda[self.mm]): # idx=0; doc=[(2,0.018),(5,0.974)]
                                                      # idx=4; doc=[(5,0.984)]
            doc_top.append([0] * num_topics)       # [[0,0,0,0,0,0,0,0,0,0]]
            for topic in doc:
                doc_top[idx][topic[0]] = topic[1]  # topic = (2, 0.0182)
                                                   # doc[idx,2] = 0.0182
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
                              reverse=True) # [3, 7, 0, 1, ...] --> (Largest) 3->7->...

            # Create a new list with the ranks.
            for rank, topic in enumerate(top_sort):
                if prob_rank[rank] > 0:
                    top_rank[topic] = rank
                else:
                    top_rank[topic] = num_topics - 1

            doc_top_rank.append(top_rank)

        return doc_top_rank

    def generate_doc_word(self, doc_top):
        top_word = self.lda.expElogbeta # topic*word 2d array matrix
        doc_word = [[0 for i in range(len(top_word[0]))] for j in range(len(doc_top))]
        for i, doc in enumerate(doc_top):
            for j, word in enumerate(top_word[0]):
                for k, topic in enumerate(doc):
                    doc_word[i][j] = doc_word[i][j] + doc_top[i][k] * top_word[k][j]

        return doc_word

    def generate_top_words(self, num_word):

        top_matrix = self.lda.expElogbeta  # top_word_matrix

    # Convert scores to rankings
        top_word_index = [[0 for i in range(len(top_matrix[0]))] for j in range(len(top_matrix))]
        for i, top_word in enumerate(top_matrix):  # CHANGE to top_matrix to test correctness
            top_word_index[i] = numpy.argsort(top_word)[::-1]
       # self.lda.expElogbeta: the topic*word distribution matrix
       # self.id2word.id2token[1]: first word in dictionary

    # Convert rankings to topic-word list
        top_word_list = []

        for i, word_list in enumerate(top_word_index):
            top_word = []
            for j, word_id in enumerate(word_list):
                if j == num_word:
                    break
                keyword_information = "%s (%f)" % (self.id2word.id2token[word_id], top_matrix[i][word_id])
                top_word.append(keyword_information) #self.id2word.id2token[word_id]) #CHANGED FOR DEBUGGING
            top_word_list.append(top_word)

        return top_word_list, top_word_index

    @staticmethod
    def gen_topic_hierarchy(topics):
        # Create the hierarchy.
        top_word_hier = {"children": []}

        for idx, word_list in enumerate(topics):
            children = []

            # Create the children nodes - words.
            top_word_str = ""
            i = 0

            for word in word_list:
                if i < 3:
                    top_word_str += word[1]
                    i += 1

                children.append({"name": word[1], "size": word[0],
                                 "value": float(word[0])*1000})
            # Add the list to current hierarchy.
            top_word_hier["children"].append({"name": "T" + str(idx), "children": children})

        top_word_hier["name"] = 'LDA Topic Modeling'
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

    @staticmethod
    def gen_topic_words(topics):
        top_word_list = []
       # top_word_index_list = []
        for idx, word_list in enumerate(topics):
            top_word = []
       #     top_word_index = []

            for word in word_list:
                top_word.append(word[1])
       #         top_word_index.append(self.lda.id2word.token2id)

            # Add the list to current hierarchy.
            top_word_list.append(top_word)
       #     top_word_index_list.append(top_word_index)

        return top_word_list  #, top_word_index_list

    """ ================ Post-processing ================ """

    def top_word_re_rank(self, num_word, re_rank_method):

        top_matrix = self.lda.expElogbeta  # top_word_matrix
        if re_rank_method == 1:
            kr_denominator = numpy.sum(top_matrix, axis=0)  # kr: keyword re-ranking

            kr_score = [[0 for i in range(len(top_matrix[0]))] for j in range(len(top_matrix))]
            for i, topic_dis in enumerate(top_matrix):
                for j, prob_word in enumerate(topic_dis):
                    kr_score[i][j] = prob_word / kr_denominator[j]  # keyword re-ranking score
        else:
            kr_denominator = scipy.stats.mstats.gmean(top_matrix, axis=0)  # kr: keyword re-ranking

            kr_score = [[0 for i in range(len(top_matrix[0]))] for j in range(len(top_matrix))]
            for i, topic_dis in enumerate(top_matrix):
                for j, prob_word in enumerate(topic_dis):
                    kr_score[i][j] = prob_word * numpy.log( prob_word / kr_denominator[j])  # keyword re-ranking score

    # Convert scores to rankings
        kr_rank = [[0 for i in range(len(top_matrix[0]))] for j in range(len(top_matrix))]
        for i, topic_score in enumerate(kr_score):  # CHANGE to top_matrix to test correctness
            kr_rank[i] = numpy.argsort(topic_score)[::-1]
       # self.lda.expElogbeta: the topic*word distribution matrix
       # self.id2word.id2token[1]: first word in dictionary

    # Convert rankings to topic-word list
        top_word_list = []

        for i, word_list in enumerate(kr_rank):
            top_word = []
            for j, word_id in enumerate(word_list):
                if j == num_word:
                    break
                keyword_information = "%s (%f -> %f)" % (self.id2word.id2token[word_id],top_matrix[i][word_id],kr_score[i][word_id])
                top_word.append(self.id2word.id2token[word_id])  #keyword_information) #CHANGED FOR DEBUGGING
            top_word_list.append(top_word)

        return top_word_list  # re_top_word

    def doc_word_index_set(self):
        index_set = [0 for i in range(len(self.mm))]
        for i, doc in enumerate(self.mm):
            index_set[i] = set()
            for word in doc:
                index_set[i].add(word[0])
        return index_set

    @staticmethod
    def topic_coherence(top_word, num_word, doc_word_index_set):
        num_topic = len(top_word)
        coherence = [0 for i in range(num_topic)]
        for t, topic in enumerate(top_word):
            for i in range(1, num_word-1):  # m
                for j in range(0, i-1):  # l
                    D = 0
                    DD = 0
                    word_m_idx = topic[i]
                    word_l_idx = topic[j]
                    for doc in doc_word_index_set:
                        if word_l_idx in doc:
                            D = D + 1
                            if word_m_idx in doc:
                                DD = DD + 1
                    coherence[t] = coherence[t] + numpy.log((DD+1)/D)

        return coherence

    @staticmethod
    def perplexity(prob_doc_word, doc_word_index_set):
        log_likelihood = 0
        total_num_word = 0
        for idx, prob_doc in enumerate(prob_doc_word):
            p_d = 1
            sorted_prob_doc = numpy.sort(prob_doc)[::-1]
            for i in range(50):
                p_d = p_d * sorted_prob_doc[i]
            log_likelihood = log_likelihood + numpy.log(p_d)
            total_num_word = total_num_word + len(doc_word_index_set[idx])
        perplexity = numpy.exp(-log_likelihood / total_num_word)
        return perplexity

    # Another tool which uses Gibbs sampling
    def mm_to_doc_word_times(self):
        doc_word_times = [[0 for i in range(len(self.id2word.token2id))] for j in range(len(self.mm))]
        for i, line in enumerate(self.mm):
            for pair in line:
                doc_word_times[i][pair[0]] = pair[1]
        doc_word_times.shape = (len(self.mm), len(self.id2word.token2id))
        return doc_word_times

 #   def train_clda(self, num_top=10, must_link, cannot_link, num_pass=1, update_t=1, chunks=2000, num_iter=50,
 #                 num_eval=10, gamma_t=0.001, kappa=0.5, tao=1.0, eta=None):

 #       alpha1 = 1 / num_top

 #       self.lda = models.LdaModel(corpus=self.mm, id2word=self.id2word,
 #                                  num_topics=num_top, alpha=alpha1,
 #                                  passes=num_pass, update_every=update_t,
 #                                  chunksize=chunks, iterations=num_iter,
 #                                  eval_every=num_eval, gamma_threshold=gamma_t,
 #                                  decay=kappa, offset=tao, eta=eta)