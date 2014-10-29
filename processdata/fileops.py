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

special_characters = [':', '@', '$', '#', '%', '^', '&', '*', '(', ')', '_', '+', '=', '-', '`'
                      , '<', '>', '?', '[', ']', '{', '}', '\\', '|', ',', '.', '!'
                      , '***************************************************************************'
                      , "''", '--']

token_list = []


class FileReader:
    def __init__(self):
        self.token_list = []

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

        try:
            file_tokens = word_tokenize(file_text)
            file_tokenized_text = Text(file_tokens)
            stop_words = stopwords.words('english')

            # Clear Stop words in the tokens and special characters.
            for token in file_tokenized_text:
                if token.lower() not in stop_words and token.lower() not in special_characters:
                    tokens.append(token.lower())

        except UnicodeDecodeError:
            print "Unicode Decode Error: Moving On"

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


"""
This function is to write to JSON file.

def write_file(data, filename):

"""