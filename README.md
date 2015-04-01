LDAExplore (CMSC691-FINALPROJECT)
====================

This is the FINAL Project code for CMSC 691.
The project uses the lda library from python which implements Latent Dirichlet Allocation (LDA) using Gibbs Sampling.

DOCUMENTATION
=============
http://radimrehurek.com/gensim/

INSTALLATION
============
 
STEPS:
1. Install pip.
2. pip install numpy, scipy, scikit-learn, gensim
3. pip install nltk
   (NLTK is used for stopwords and some other stuff)

EXAMPLE
=======

```python
from processdata import fileops
from processdata.lda import LDAVisualModel

word_corpus = fileops.read_file('20_newsgroups/alt.atheism/53350')
lda = LDAVisualModel([word_corpus])
lda.create_word_corpus([word_corpus])
lda.train_lda(3)
topics = lda.get_lda_corpus()

print topics

------
RESULT
------

topic: <word distribution>
```
(SOURCE: From the documentation, so we dont have to go it again.)

DATA
====

The Reuters 20 newsgroup data set has been added to the repository.

PAPERS
======

```latex
1. Latent Dirchlet Allocation
   Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." the Journal of machine Learning research 3 (2003): 993-1022.

2. Probabistic Topic Models
   Blei, David M. "Probabilistic topic models." Communications of the ACM 55.4 (2012): 77-84.

3. Optimizing temporal topic segmentation for intelligent text visualization
   Pan, Shimei, et al. "Optimizing temporal topic segmentation for intelligent text visualization." Proceedings of the 2013 international conference on Intelligent user interfaces. ACM, 2013.

4. Active Learning with Constrained Topic Model
   Yang, Yi, et al. "Active Learning with Constrained Topic Model."

5. Visualizing Sets and Set-typed Data: State-of-the-Art and Future Challenges 
   Alsallakh, Bilal, et al. "Visualizing Sets and Set-typed Data: State-of-the-Art and Future Challenges (Supplementary Material)."
```

