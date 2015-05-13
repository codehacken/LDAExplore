"""
Creating a simple web service to handle requests from Javascript
to perform LDA.

Using Flask package to perform LDA.
"""
__author__ = 'ashwin'

#!/usr/bin/env python
from flask import Flask, url_for
from ldaops import perform_lda

app = Flask(__name__)


welcome_message = "<html><body><p>REST API for LDAExplore</p>\n Use: /lda to perform LDA.</body></html>"

# A standard welcome message.
@app.route('/')
def api_root():
    return welcome_message

# API to perform LDA.
@app.route('/lda')
def api_lda():
    perform_lda()


if __name__ == "__main__":
    app.run()
