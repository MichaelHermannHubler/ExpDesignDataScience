# imports 

import pandas as pd
import numpy as np
from sklearn import datasets
import os
import nltk
from nltk.corpus import stopwords

# run in normal python shell once:
#>>> import nltk
#>>> nltk.download()
# and download all packages (easiest)

# Tokenize
def tokenize_text (text):
    tokens = []
    for sentences in nltk.sent_tokenize(text):
        for token in nltk.word_tokenize(text):
            if token not in stopwords.words('english'):
                tokens.append(token)
    return tokens

def get_20_newsgroups_dataset():    
    # 20 newsgroups dataset

    newsgroups_train = datasets.fetch_20newsgroups(subset='train')
    newsgroups_test = datasets.fetch_20newsgroups(subset='test')

    return (newsgroups_train, newsgroups_test)

def get_subjectivity_dataset():
    # Subjectivity dataset
    subjective_sentences_file = './data/Movie_Review/Subjectivity/plot.tok.gt9.5000'
    objective_sentences_file = './data/Movie_Review/Subjectivity/quote.tok.gt9.5000'

    subjective_sentences = pd.read_csv(subjective_sentences_file, header=None, sep='|', encoding='latin-1', names=['text'])
    objective_sentences = pd.read_csv(objective_sentences_file, header=None, sep='|', encoding='latin-1', names=['text'])

    subjective_sentences['subjective'] = 1
    objective_sentences['subjective'] = 0

    subjectivity_complete = pd.concat([subjective_sentences, objective_sentences])
    
    # Tokenize Subjectivity
    subjectivity_tokenized = [tokenize_text(text) for text in subjectivity_complete['text']]   # Tokenized data
    subjectivity_complete['tokens'] = subjectivity_tokenized

    return subjectivity_complete

def get_polarity_dataset():
    # Sentence polarity dataset

    neg_directory = './data/Movie_Review/Sentence_Polarity/neg/'
    pos_directory = './data/Movie_Review/Sentence_Polarity/pos/'

    neg_files = os.listdir(neg_directory)
    pos_files = os.listdir(pos_directory)

    neg_sentences = []
    pos_sentences = []

    for file in neg_files:
        with open(neg_directory + file, 'r') as f:
            neg_sentences.append(f.read())
            
    for file in pos_files:
        with open(pos_directory + file, 'r') as f:
            pos_sentences.append(f.read())

    neg_sentences = pd.DataFrame(neg_sentences, columns=['text'])
    pos_sentences = pd.DataFrame(pos_sentences, columns=['text'])

    neg_sentences['positive'] = 0
    pos_sentences['positive'] = 1

    polarity_complete = pd.concat([neg_sentences, pos_sentences])

    # Tokenize Polarity
    polarity_tokenized = [tokenize_text(text) for text in polarity_complete['text']]   # Tokenized data
    polarity_complete['tokens'] = polarity_tokenized

    return polarity_complete