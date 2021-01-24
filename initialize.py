# imports 

import pandas as pd
import numpy as np
from sklearn import datasets
import os
import nltk
from nltk.corpus import stopwords
import pickle
import urllib.request

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

    if os.path.isfile('./data/variable_storage/polarity.pkl'):
        return pickle.load(open('./data/variable_storage/polarity.pkl', 'rb'))
    else :
        print('Beginning file download with urllib2...')
        url = 'https://waps.hermann-hubler.com/Download/Data/polarity.pkl'
        urllib.request.urlretrieve(url, './data/variable_storage/polarity.pkl')
        return pickle.load(open('./data/variable_storage/polarity.pkl', 'rb'))

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

def get_TREC45_dataset():
    trec_file = './data/TREC45/04.testset'

    trec = []
    trec_lines = []

    #load all text
    with open(trec_file, 'r') as f:     
        line = f.readline()
        while line:
            if line != '':
                if line.startswith('<'):
                    trec_lines.append(line.strip())
                else:
                    trec_lines[-1] = trec_lines[-1] + ' ' + line.strip()

            line = f.readline()
    
    # transform text into dataframe
    trec_elem = [None] * 4
    for i in range(len(trec_lines)):
        if trec_lines[i].startswith('<top>'):
            trec_elem = [None] * 4
        elif trec_lines[i].startswith('</top>'):
            trec.append(trec_elem)
        elif trec_lines[i].startswith('<num>'):
            trec_elem[0] = trec_lines[i][5:]
        elif trec_lines[i].startswith('<title>'):
            trec_elem[1] = trec_lines[i][7:]
        elif trec_lines[i].startswith('<desc>'):
            trec_elem[2] = trec_lines[i][6:]
        elif trec_lines[i].startswith('<narr>'):
            trec_elem[3] = trec_lines[i][6:]

    trec = pd.DataFrame(trec, columns=['Number', 'Title', 'Description', 'Narrative'])

    # remove identifying Text from content
    trec['Number'] = trec['Number'].map(lambda x: x.lstrip('Number: '))
    trec['Description'] = trec['Description'].map(lambda x: x.lstrip('Description: '))
    trec['Narrative'] = trec['Narrative'].map(lambda x: x.lstrip('Narrative: '))

    # tokenize
    trec['Title_tokenized'] = [tokenize_text(text) for text in trec['Title']]   # Tokenized data
    trec['Description_tokenized'] = [tokenize_text(text) for text in trec['Description']]   # Tokenized data
    trec['Narrative_tokenized'] = [tokenize_text(text) for text in trec['Narrative']]   # Tokenized data

    return trec