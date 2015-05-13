__author__ = 'Ashwin'
__email__ = 'gashwin1@umbc.edu'

"""
Perform basic file operations that can be used to feed the corpus into
other models such as LDA.

The module uses NLTK's english language tokenizer and stop word
list to clear the document's and generate a set of tokens.
"""

'''
Using nltk to clear stopwords from a document.
'''
from nltk import word_tokenize, Text
from nltk.corpus import stopwords

import glob
import re
import csv
import json

regex_clear = "^[a-zA-Z0-9_@]*$"


# This is the class to work with CSV files.
def read_csv(filename, delimit='"', quote='|'):
    file_ptr = open(filename, 'r')
    csv_reader = csv.reader(file_ptr, delimiter=delimit,
                            quotechar=quote)

    # Use a generator to provide lines.
    for row in csv_reader:
        yield row


class FileReader:
    def __init__(self):
        self.token_list = []

    def read_file_text(self, filename):
        file_handle = open(filename, "r")

        for line in file_handle:
            self.token_list.append(line) # token_list = ['string_a', 'string_b']

        file_handle.close()

    def read_data_dir_loc(self, dir_name):
        filenames = glob.glob("%s/*" % dir_name)
        for filename in filenames:
            file_handle = open(filename, "r")
            file = file_handle.read().replace('\n', ' ')
            self.token_list.append(file) # token_list = ['string_a', 'string_b']
            file_handle.close()

    # This is a specific function which reads a single file but treats every line as
    # document.
    # E.g. Abstracts from different documents can be treated as a separate document for
    #      each abstract.

    def read_text_sections(self, filename, s_words=[], e_words=[]):

        file_handle = open(filename, "r")

        i = 0
        for line in file_handle:
            #try:
            tokens = []
            file_tokens = word_tokenize(line)  # Convert to separated words and signs
            file_tokenized_text = Text(file_tokens)  # Reconstruct them without signs
            stop_words = stopwords.words('english')

            # Clear Stop words in the tokens and special characters.
            for token in file_tokenized_text:
                lower_str = token.lower()
                if (lower_str not in stop_words) and (re.match(regex_clear, lower_str)) and (len(lower_str) > 2)\
                        and (not(lower_str.isdigit())) \
                        and (lower_str not in e_words):
                    tokens.append(lower_str)
# and (lower_str not in s_words[i])
            #except UnicodeDecodeError:
            #    print "Unicode Decode Error: Moving On"

            if len(tokens) != 0:
                self.token_list.append(tokens)

            i += 1

        file_handle.close()

    def read_text_sections_dir(self, dir_name, s_words=[], e_words=[]):
        filenames = glob.glob('*')
        file_handle = open(filename, "r")

        i = 0
        for line in file_handle:
            #try:
            tokens = []
            file_tokens = word_tokenize(line)  # Convert to separated words and signs
            file_tokenized_text = Text(file_tokens)  # Reconstruct them without signs
            stop_words = stopwords.words('english')

            # Clear Stop words in the tokens and special characters.
            for token in file_tokenized_text:
                lower_str = token.lower()
                if (lower_str not in stop_words) and (re.match(regex_clear, lower_str)) and (len(lower_str) > 2)\
                        and (not(lower_str.isdigit())) \
                        and (lower_str not in e_words):
                    tokens.append(lower_str)

            if len(tokens) != 0:
                self.token_list.append(tokens)

            i += 1

        file_handle.close()

    def read_file(self, filename):
        """
        This function reads a file and returns a set of tokens back.
        :param filename: This is name of file to be read.
        """
        tokens = []

        file_handle = open(filename, "r")
        file_text = file_handle.read()

        # file_text contains the whole file.
        # This is used because the current file contents are not large
        # although the number of files are large in number.

        #try:
        file_tokens = word_tokenize(file_text)
        file_tokenized_text = Text(file_tokens)
        stop_words = stopwords.words('english')

        # Clear Stop words in the tokens and special characters.
        for token in file_tokenized_text:
            lower_str = token.lower()
            if lower_str not in stop_words and re.match(regex_clear, lower_str) and len(lower_str) > 2\
                    and not(lower_str.isdigit()):
                tokens.append(lower_str)

        #except UnicodeDecodeError:
        #    print "Unicode Decode Error: Moving On"

        file_handle.close()
        if len(tokens) != 0:
            self.token_list.append(tokens)

    def read_dir(self, file_dir_name):
        """
        This function reads a directory of files and returns a list of
        token lists.
        :param file_dir_name: This is the name of the directory.
        """

        files = glob.glob(file_dir_name+"/*")
        for file_name in files:
            self.read_file(file_name)

    def get_token_list(self):
        return self.token_list


# This function is to writes to a CSV file.
# The file contains the probability of each topic.
def write_prob_to_file(doc_to_word, doc_top, num_of_words, num_topics, t_file, filename):
    # Write the headers for the columns to the CSV.
    col_string = "name,group,"
    for i in range(0, num_topics - 1):
        col_string += "T" + str(i+1) + ","
    col_string += "T" + str(num_topics) + ",ID\n"
    # Write the document information to the CSV file.
    csvreader = read_csv(t_file)

    # Write the document information to the CSV file.
    for idx, doc in enumerate(doc_top):

        # This is the name of the document.
        doc_string = csvreader.next()[0]

        # Write the title to the document.
        col_string += "\"TITLE: " + doc_string + " WORDS: "
        for i in range(0, num_of_words - 1):
            col_string += str(doc_to_word[idx][i][0]) + ", "
        col_string += str(doc_to_word[idx][i+1][0]) + "\","

        # Write the title of the document.
        col_string += doc_string

        #col_string += "D" + str(idx)
        for topic in doc:
            col_string += "," + str(topic)
        col_string += "," + str(idx) + "\n"

    with open(filename, "w") as file_handle:
        file_handle.write(col_string)


# This function is to writes to a CSV file.
# The file contains the rank of each topic.
def write_rank_to_file(doc_to_word, doc_top_rank, num_of_words, num_topics, t_file, d_file):
    # Write the headers for the columns to the CSV.
    col_string = "name,group,"
    for i in range(0, num_topics - 1):
        col_string += "T" + str(i+1) + ","
    col_string += "T" + str(num_topics) + ",ID\n"

    # Write the document information to the CSV file.
    csvreader = read_csv(t_file)
    for idx, doc in enumerate(doc_top_rank):

        # This is the name of the document.
        doc_string = csvreader.__next__()[0]

        # Write the title to the document.
        col_string += "\"TITLE: " + doc_string + " WORDS: "
        for i in range(0, num_of_words - 1):
            col_string += str(doc_to_word[idx][i][0]) + ", "
        col_string += str(doc_to_word[idx][i+1][0]) + "\","

        # Construct the Ranking for each topic.
        # Make all the topics that have prob. 0 as the last rank.
        #col_string += "D" + str(idx+1)
        col_string += doc_string
        for topic in doc:
            col_string += "," + str(topic+1)
        col_string += "," + str(idx+1) + "\n"

    # Final writing to the document.
    with open(d_file, "w") as file_handle:
        file_handle.write(col_string)


# The file written contains the hierarchial structure of words and topics.
def write_top_hier_to_file(top_hier, filename):
    with open(filename, "w") as file_ptr:
        json.dump(top_hier, file_ptr)


def write_top_word_to_file(top_word, filename, top_coherence, coherence, perplexity):
    write_string = "Coherence: %f  Perplexity: %f \n" % (coherence, perplexity)

    for i, topic in enumerate(top_word):
        write_string += "T" + str(i+1) + " (" + str(top_coherence[i]    ) + ") : "
        for word in topic:
            write_string += '"' + str(word) + '", '
        write_string += "\n"

    with open(filename, "w") as file_ptr:
        file_ptr.write(write_string)


def write_top_word_matrix_to_file(word_dict, matrix, filename):
    write_string = ","
    for idx in word_dict:
        write_string += word_dict[idx] + ','
    write_string += '\n'

    for idx, topic_prob in enumerate(matrix):
        write_string += 'T' + str(idx+1) + ','
        for word_prob in topic_prob:
            write_string += "%.20f" % word_prob + ','
        write_string += '\n'

    with open(filename, "w") as file_ptr:
        file_ptr.write(write_string)


def write_evaluation_matrix_to_file(coherences, perplexities, filename):
    write_string = ""
    for coherence in coherences:
        write_string += str(coherence) + ' '
    write_string += '\n'
    for perplexity in perplexities:
        write_string += str(perplexity) + ' '
    write_string += '\n'

    with open(filename, "w") as file_ptr:
        file_ptr.write(write_string)