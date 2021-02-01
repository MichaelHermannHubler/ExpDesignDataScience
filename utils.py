import os
import logging
import pandas as pd
import numpy as np
from numpy import random
from random import shuffle

import gensim
from gensim import matutils
from gensim import corpora
from gensim.models import LsiModel, Word2Vec, Doc2Vec, KeyedVectors
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.doc2vec import TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

import re

import sys
import csv
#csv.field_size_limit(sys.maxsize)
csv.field_size_limit(2147483647)

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from spherecluster import VonMisesFisherMixture



### Tokenizer ###

clean = lambda t: re.sub('[,?;*!%^&_+():-\[\]{}]', ' ', t.replace('"', ' ').replace('/', ' ').replace('\\', ' ').replace("'", ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('-', ' ').replace('.', '').replace('&hyph;', ' ').replace('&blank;', ' ').replace('  ', ' ').strip().lower())

def tokenize_text(text):
    """Tokenize a document.
    
    Arguments:
        text -- text of a single document; type: .
    
    Returns:
        tokens -- tokenized text; type: list of strings.
    """
    
    tokens = []
    
    for sent in nltk.sent_tokenize(text, language='english'):  # split string into sentences
        for word in nltk.word_tokenize(sent, language='english'):  # split each sentence into words (tokens)
            if len(word) < 2:  # remove short words
                continue
            if word in stopwords.words('english'):  # remove stopwords
                continue
            tokens.append(word)  # downcase
    return tokens



def identity_tokenizer(text):
    """Return text as it is."""
    return text



### Model evaluation ###

def evaluate_prediction_BoW(vectorizer, classifier, test_data):
    """Evaluate classification accuracy of the trained models.
    
    Arguments:
        vectorizer -- 
        classifier -- 
        test_data -- 

    Returns:
        accuracy_score -- classification accuracy; type: float.
    """
    
    data = (test_data[k][0] for k in range(len(test_data)))  # generator for the train data
    data_features = vectorizer.transform(data)
    predictions = classifier.predict(data_features)
    target = [test_data[k][1] for k in range(len(test_data))]
    
    return accuracy_score(target, predictions)



def evaluate_prediction(classifier, test_data, labels):
    """Evaluate classification accuracy of the trained models.
    
    Arguments:
        classifier -- here logistic regr. classifier.
        test_data -- test dataset.
        labels -- ground truth labels.

    Returns:
        accuracy_score -- classification accuracy; type: float.
    """
    
    predictions = classifier.predict(test_data)
    
    return accuracy_score(labels, predictions)



def evaluate_cluster(cluster_model, labels):
    """Evaluate clustering performance of the models.
    
    Arguments:
        cluster_model -- here logistic regr. classifier.
        labels -- ground truth labels.

    Returns:
        ARI, NMI -- Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) metrics; 
                    type: float.
    """
    
    predictions = cluster_model.labels_  # Predict labels

    ARI = adjusted_rand_score(labels, predictions)
    NMI = normalized_mutual_info_score(labels, predictions)
    
    return ARI, NMI



### General utils ###

class LossLogger(CallbackAny2Vec):
    '''Get the Loss after every epoch and log it.'''
    
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logging.info('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1
        
        
        
def BoWE_doc(wv, words):
    """BoWE representation of a document, i.e. map words to word vectors:
    X = {w1, ..., wT} --> E_X = {Ew1, ..., EwT}.
    
    Arguments:
        wv -- word vector model.
        words -- document text.

    Returns:
        BoWE -- BoWE representation of a document X; type: list of floats.
    """
    
    BoWE = []
    
    for word in words:
        if isinstance(word, np.ndarray):
            BoWE.append(word)
        elif word in wv.vocab:
            BoWE.append(wv.word_vec(word))
            
    if not BoWE:
        logging.warning("Cannot compute similarity with no input: %s", words)
        # Remove these examples in pre-processing...
        return np.zeros(50,)

    BoWE = [gensim.matutils.unitvec(BoWE[j]).astype(np.float32) for j in range(len(BoWE))]
    
    return BoWE



def word_averaging(wv, words):
    """Average word vectors to get a fixed-length representation
    of documents.
    
    Arguments:
        wv -- word vector model.
        words -- document text.

    Returns:
        mean -- averaged word vector representation of a document; 
                type: list of floats.
    """
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.vectors_norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("Cannot compute similarity with no input: %s", words)
        # Remove these examples in pre-processing...
        return np.zeros(50,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    
    return mean



def word_averaging_list(wv, text_list):
    """Averaged word vector representation of a set of documents.
    
    Arguments:
        wv -- word vector model.
        text_list -- list of documents.

    Returns:
        feature matrix containg the averaged word vector representation
        of each document; type: list of floats; shape: # docs. x 50. 
    """
    return np.vstack([word_averaging(wv, review) for review in text_list])



### k-fold cross-validation ###

def chunks(data, n):
    """ Yield n successive chunks from an array `data`.
    
    Arguments:
        data -- data to split in train/test sets; type: list.
        n -- number of folds; type: integer.

    Returns:
        train_chunk, test_chunk -- train and test sets.
    """
    newn = int(len(data) / n)  # chunk size 
    
    for i in range(0, n-1):
        test_chunk = data[i*newn:i*newn+newn]
        train_chunk = [el for el in data if el not in test_chunk]
        yield train_chunk, test_chunk
        
    test_chunk = data[n*newn-newn:]
    train_chunk = [el for el in data if el not in test_chunk]
    
    yield train_chunk, test_chunk

    
    
def k_fold_BoW(data, vectorizer, features, k=10, reg=1):
    """Performs k-fold cross-validation on the dataset `data` using BoW features.
    
    Arguments:
        data -- tokenized data; type: list of strings.
        vectorizer -- vectorizer to be used; possibilities: CountVectorizer, TfidfVectorizer.
        features -- number of top features to keep; type: int.
        k -- number of folds for the k-fold cross-eval.; type: int.
        reg -- regularization parameter for logistic regr.; type: positive float.
    
    Returns:
        k_fold_acc -- list with the accuracy on the k folds; type: list of floats.
    """
    
    shuffle(data)  # random shuffle data before making folds
    
    folds = chunks(data, k)
    k_fold_acc = []
    
    for fold in folds:
        # CountVectorizer: convert a collection of text documents 
        # to a matrix of token counts.
        count_vectorizer = vectorizer(tokenizer=identity_tokenizer, lowercase=False,
                                      max_features=features) 
        
        # Matrix of shape len(subj_train) x #words.
        train_data = (fold[0][k][0] for k in range(len(fold[0])))  # text for the training data
        train_features = count_vectorizer.fit_transform(train_data)

        ### Logistic regression classifier.
        logreg = linear_model.LogisticRegression(C=reg, n_jobs=1, solver='liblinear', multi_class='ovr')
        train_tag = [fold[0][k][1] for k in range(len(fold[0]))]  # labels for the trainig data
        logreg = logreg.fit(train_features, train_tag)
        
        test_data = fold[1]  # Both text and labels
        acc = evaluate_prediction_BoW(count_vectorizer, logreg, test_data)
        k_fold_acc.append(acc)
        
    return k_fold_acc



def k_fold_LSI(data, topics=50, k=10, reg=1):
    """Performs k-fold cross-validation on the dataset `data` using LSI features.
    
    Arguments:
        data -- tokenized data; type: list of strings.
        topics -- number of latent topics to use; type: int.
        k -- number of folds for the k-fold cross-eval.; type: int.
        reg -- regularization parameter for logistic regr.; type: positive float.
    
    Returns:
        k_fold_acc -- list with the accuracy on the k folds; type: list of floats.
    """
    
    tokenized_data_text = [data[k][0] for k in range(len(data))]  # data

    # Create a Gensim dictionary and corpus.
    dct = corpora.Dictionary(tokenized_data_text)
    # Gensim uses bag of wards to represent in this form.
    corpus = [dct.doc2bow(sent) for sent in tokenized_data_text] 
    
    # Run LSI model to get topic modelling.
    lsi_model = LsiModel(corpus=corpus, id2word=dct, num_topics=topics)
    
    # Converting topics to feature vectors - the probability distribution 
    # of the topics over a specific review is our feature vector.        
    lsi_corp = lsi_model[corpus]
    feat_vecs = matutils.corpus2dense(lsi_corp, num_terms=topics).T.tolist()    
        
    # Create train/test sets.
    # Random shuffle data (LSI feature vectors in `feat_vecs`)
    # and tags in the same way.
    data_tags = [data[k][1] for k in range(len(data))]  # tags
    comb_data = list(zip(feat_vecs, data_tags))
    random.shuffle(comb_data)
    folds = chunks(comb_data, k)
    
    k_fold_acc = []
    
    for fold in folds:
        # Training data
        X_train = [fold[0][k][0] for k in range(len(fold[0]))]  # text 
        y_train = [fold[0][k][1] for k in range(len(fold[0]))]  # labels
        
        # Test data
        X_test = [fold[1][k][0] for k in range(len(fold[1]))]  # text 
        y_test = [fold[1][k][1] for k in range(len(fold[1]))]  # labels
        
        ## Logistic regression classifier.

        # Use the elements in train_vecs as feature vectors.
        logreg = linear_model.LogisticRegression(C=reg, n_jobs=1, solver='liblinear', multi_class='ovr')
        logreg = logreg.fit(X_train, y_train)
        
        ## Evaluation.
        acc = evaluate_prediction(logreg, X_test, y_test)
        k_fold_acc.append(acc)
        
    return k_fold_acc



def k_fold_LDA(data, save_path=None, name=None, topics=50, passes=30, chunksize=1000, k=10, reg=1):
    """Performs k-fold cross-validation on the dataset `data` using LDA features.
    
    Arguments:
        data -- tokenized data; type: list of strings.
        save_path -- path where the model is saved; type: string.
        name -- dataset name; type: string; possibilities: `subj`, `sent`.
        topics -- number of latent topics to use; type: int.
        passes -- number of passes through the training set for training; type: int.
        chunksize -- chunksize used for training; type: int.
        k -- number of folds for the k-fold cross-eval.; type: int.
        reg -- regularization parameter for logistic regr.; type: positive float.
    
    Returns:
        k_fold_acc -- list with the accuracy on the k folds; type: list of floats.
    """
    
    tokenized_data_text = [data[k][0] for k in range(len(data))]  # data

    # Create a Gensim dictionary and corpus.
    dct = corpora.Dictionary(tokenized_data_text)
    # Gensim uses bag of wards to represent in this form.
    corpus = [dct.doc2bow(sent) for sent in tokenized_data_text] 
    
    # Run LDA model to get topic modelling.
    lda_model = LdaMulticore(corpus=corpus, id2word=dct, num_topics=topics, passes=passes,
                             chunksize=chunksize, eval_every=1000, workers=4)
    
    # Save model to disk (optional).
    if save_path is not None:
        save_dir = os.path.join(save_path, "lda_{0}_passes{1}_reg{2:.3f}.model".format(name, passes, reg))
        lda_model.save(save_dir)
    
    # Converting topics to feature vectors - the probability distribution 
    # of the topics over a specific review is our feature vector.
    feat_vecs = []  # store the feature vectors of size `topics`

    for i in range(len(corpus)):
        top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[j][1] for j in range(topics)]
        feat_vecs.append(topic_vec)
    
    # Create train/test sets.
    # Random shuffle data (LDA feature vectors in `feat_vecs`)
    data_tags = [data[k][1] for k in range(len(data))]  # tags
    comb_data = list(zip(feat_vecs, data_tags))
    random.shuffle(comb_data)
    folds = chunks(comb_data, k)
    
    k_fold_acc = []
    
    for fold in folds:
        # Training data
        X_train = [fold[0][k][0] for k in range(len(fold[0]))]  # text 
        y_train = [fold[0][k][1] for k in range(len(fold[0]))]  # labels
        
        # Test data
        X_test = [fold[1][k][0] for k in range(len(fold[1]))]  # text 
        y_test = [fold[1][k][1] for k in range(len(fold[1]))]  # labels
        
        ## Logistic regression classifier.

        # Use the elements in train_vecs as feature vectors.
        logreg = linear_model.LogisticRegression(C=reg, n_jobs=1, solver='liblinear', multi_class='ovr')
        logreg = logreg.fit(X_train, y_train)
        
        ## Evaluation.
        acc = evaluate_prediction(logreg, X_test, y_test)
        k_fold_acc.append(acc)
        
    return k_fold_acc



def k_fold_cBow(data, save_path=None, name=None, v_size=50, epochs=30, k=10, reg=1):
    """Performs k-fold cross-validation on the dataset `data` using cBow (Word2vec) features.
    
    Arguments:
        data -- tokenized data; type: list of strings.
        save_path -- path where the model is saved; type: string.
        name -- dataset name; type: string; possibilities: `subj`, `sent`.
        v_size -- dimensionality of the word vectors; type: int.
        epochs -- number of training epochs; type: int.
        k -- number of folds for the k-fold cross-eval.; type: int.
        reg -- regularization parameter for logistic regr.; type: positive float.
    
    Returns:
        k_fold_acc -- list with the accuracy on the k folds; type: list of floats.
    """
    
    tokenized_data_text = [data[k][0] for k in range(len(data))]  # data
    
    ### Train Word2Vec ###
    cores = 4      # Threads used for training

    # Initialize the model
    w2v_model = Word2Vec(size=v_size, window=5, min_count=1, workers=cores)

    # Build the vocabulary
    w2v_model.build_vocab(tokenized_data_text, progress_per=5000)

    # Train the model
    w2v_model.train(tokenized_data_text, total_examples=w2v_model.corpus_count,
                    epochs=epochs, compute_loss=True, callbacks=[LossLogger()])
    
    # Save model to disk (optional).
    if save_path is not None:
        save_dir = os.path.join(save_path, "word2vec_{0}_epochs{1}_reg{2:.3f}.model".format(name, epochs, reg))
        w2v_model.save(save_dir)
    
    w2v_model.init_sims(replace=True)
    
    # Create train/test sets.
    data_tags = [data[k][1] for k in range(len(data))]  # tags
    comb_data = list(zip(tokenized_data_text, data_tags))
    random.shuffle(comb_data)
    folds = chunks(comb_data, k)
        
    k_fold_acc = []
    
    for fold in folds:
        # Training data
        X_train = [fold[0][k][0] for k in range(len(fold[0]))]  # text 
        y_train = [fold[0][k][1] for k in range(len(fold[0]))]  # labels
        
        # Test data
        X_test = [fold[1][k][0] for k in range(len(fold[1]))]  # text 
        y_test = [fold[1][k][1] for k in range(len(fold[1]))]  # labels
        
        # Get sentence embedding by averaging the words vectors.
        X_train_aver = word_averaging_list(w2v_model.wv, X_train)
        X_test_aver = word_averaging_list(w2v_model.wv, X_test)
        
        ## Logistic regression classifier.

        # Use the elements in train_vecs as feature vectors.
        logreg = linear_model.LogisticRegression(C=reg, n_jobs=1, solver='liblinear', multi_class='ovr')
        logreg = logreg.fit(X_train_aver, y_train)
        
        ## Evaluation.
        acc = evaluate_prediction(logreg, X_test_aver, y_test)
        k_fold_acc.append(acc)
        
    return k_fold_acc



def k_fold_PV(data, save_path=None, name=None, v_size=50, epochs=[30], k=10, reg=1):
    """Performs k-fold cross-validation on the dataset `data` using PV (Doc2vec) features.
    
    Arguments:
        data -- tokenized data; type: list of strings.
        save_path -- path where the model is saved; type: string.
        name -- dataset name; type: string; possibilities: `subj`, `sent`.
        v_size -- dimensionality of the feature vectors; type: int.
        epochs -- number of training epochs for DBOW and DM; type: list;
                  e.g. [epochs_DBOW, epochs_DM].
        k -- number of folds for the k-fold cross-eval.; type: int.
        reg -- regularization parameter for logistic regr.; type: positive float.
    
    Returns:
        k_fold_acc -- list with the accuracy on the k folds; type: list of floats.
    """
	
    tokenized_data_text = [data[k][0] for k in range(len(data))]  # data
    
    # For Doc2Vec data need to be tokenized + tagged.
    tagged_data_text = []

    for j, sent in enumerate(tokenized_data_text):
        tagged_data_text.append(TaggedDocument(words=sent, tags=[j]))
    
    ### Train Doc2Vec ###
    assert gensim.models.doc2vec.FAST_VERSION > -1, "Too slow otherwise"
    
    cores = 4      # Threads used for training

    # Initialize 2 models: PV-DBOW and PV-DM.
    d2v_models = [
        # PV-DBOW (dm=0).
        Doc2Vec(dm=0, vector_size=v_size, window=5, min_count=1, sample=0, workers=cores),
        # PV-DM (dm=1) with default averaging.
        Doc2Vec(dm=1, vector_size=v_size, window=5, min_count=1, sample=0, workers=cores)
    ]

    # Build the vocabulary
    for model in d2v_models:
        model.build_vocab(tagged_data_text)
        print("%s Vocabulary ready for " % model)
        
    # Train the models.
    for j, model in enumerate(d2v_models): 
        print("Training %s" % model)
        
        if len(epochs) == 1:  # Train both models for the same epoch number.
            model.train(tagged_data_text, total_examples=model.corpus_count, epochs=epochs[0])
        else:  # Two different numbers of epochs.
            model.train(tagged_data_text, total_examples=model.corpus_count, epochs=epochs[j])
        
        # Save model to disk (optional).
        if save_path is not None:
            if len(epochs) == 1:
                save_dir = os.path.join(save_path, "doc2vec_{0}_epochs{1}_reg{2:.3f}.model".format(name, epochs[0], reg))
            else:
                save_dir = os.path.join(save_path, "doc2vec_{0}_epochs{1}_reg{2:.3f}.model".format(name, epochs[j], reg))
            model.save(save_dir)
        
        model.init_sims(replace=True)
    
    # Create train/test sets.
    data_tags = [data[k][1] for k in range(len(data))]  # tags
    comb_data = list(zip(tagged_data_text, data_tags))
    random.shuffle(comb_data)
    folds = chunks(comb_data, k)
        
    k_fold_acc = [[], []]  # First element: DBOW, second: DM.
    
    for fold in folds:
        # Training data
        X_train = [fold[0][k][0] for k in range(len(fold[0]))]  # text 
        y_train = [fold[0][k][1] for k in range(len(fold[0]))]  # labels
        
        # Test data
        X_test = [fold[1][k][0] for k in range(len(fold[1]))]  # text 
        y_test = [fold[1][k][1] for k in range(len(fold[1]))]  # labels
        
        for j, model in enumerate(d2v_models):
            # Infering the train vectors before training the classifier.
            X_train_vec = [model.infer_vector(doc.words) for doc in X_train]
    
            ## Logistic regression classifier.
            logreg = linear_model.LogisticRegression(n_jobs=1, C=reg, solver='liblinear', multi_class='ovr')
            logreg = logreg.fit(X_train_vec, y_train)
    
            # Infering the test vectors.
            X_test_vec = [model.infer_vector(doc.words) for doc in X_test]
    
            ## Evaluation.
            acc = evaluate_prediction(logreg, X_test_vec, y_test)
            k_fold_acc[j].append(acc)
        
    return k_fold_acc[0], k_fold_acc[1]



def k_fold_FVGMM(data, wv_model, n_comp=15, k=10, reg=1):
    """Performs k-fold cross-validation on the dataset `data` using FV-GMM features.
    
    Arguments:
        data -- tokenized data; type: list of strings.
        wv_model -- word vector model; type: Gensim model.
        n_comp -- number of mixture components; type: int.
        k -- number of folds for the k-fold cross-eval.; type: int.
        reg -- regularization parameter for logistic regr.; type: positive float.
    
    Returns:
        k_fold_acc -- list with the accuracy on the k folds; type: list of floats.
    """
    
    ## Prepare the corpus.
    tokenized_data_text = [data[k][0] for k in range(len(data))]  # data
    
    # Initialize a GMM with K components.
    gmm_neu = mixture.GaussianMixture(n_components=n_comp, covariance_type='diag', 
                                      max_iter=300, n_init=10, reg_covar=1e-05)
    
    # Fit the word embedding data with the GMM model.
    gmm_neu.fit(wv_model.vectors)
    
    ## Create train/test sets.
    data_tags = [data[k][1] for k in range(len(data))]  # tags
    comb_data = list(zip(tokenized_data_text, data_tags))
    random.shuffle(comb_data)
    folds = chunks(comb_data, k)
        
    k_fold_acc = []
    
    for fold in folds:
        # Training data
        X_train = [fold[0][k][0] for k in range(len(fold[0]))]  # text 
        y_train = [fold[0][k][1] for k in range(len(fold[0]))]  # labels
        
        # Test data
        X_test = [fold[1][k][0] for k in range(len(fold[1]))]  # text 
        y_test = [fold[1][k][1] for k in range(len(fold[1]))]  # labels
        
        # Get sentence embedding by using the FVs.
        X_train_FV = [FV_GMM(BoWE_doc(wv_model, X_train[k]), gmm_neu) for k in range(len(X_train))]
        X_test_FV = [FV_GMM(BoWE_doc(wv_model, X_test[k]), gmm_neu) for k in range(len(X_test))]
        
        ## Logistic regression classifier.

        # Use the elements in train_vecs as feature vectors.
        logreg = linear_model.LogisticRegression(C=reg, n_jobs=1, solver='liblinear', multi_class='ovr')
        logreg = logreg.fit(X_train_FV, y_train)

        ## Evaluation.
        acc = evaluate_prediction(logreg, X_test_FV, y_test)
        k_fold_acc.append(acc)
        
    return k_fold_acc



def k_fold_FVmoVMF(data, wv_model, n_comp=15, k=10, reg=1):
    """Performs k-fold cross-validation on the dataset `data` using FV-moVMF features.
    
    Arguments:
        data -- tokenized data; type: list of strings.
        wv_model -- word vector model; type: Gensim model.
        n_comp -- number of mixture components; type: int.
        k -- number of folds for the k-fold cross-eval.; type: int.
        reg -- regularization parameter for logistic regr.; type: positive float.
    
    Returns:
        k_fold_acc -- list with the accuracy on the k folds; type: list of floats.
    
    """
    
    ## Prepare the corpus.
    tokenized_data_text = [data[k][0] for k in range(len(data))]  # data
    
    # Initialize a moVMF with K components.
    vmf_neu = VonMisesFisherMixture(n_clusters=n_comp, posterior_type='soft', n_init=4, n_jobs=-2,
                                    init='k-means++')

    # Fit the word embedding data with the GMM model.
    vmf_neu.fit(normalize(wv_model.vectors))
    
    ## Create train/test sets.
    data_tags = [data[k][1] for k in range(len(data))]  # tags
    comb_data = list(zip(tokenized_data_text, data_tags))
    random.shuffle(comb_data)
    folds = chunks(comb_data, k)
        
    k_fold_acc = []
    
    for fold in folds:
        # Training data
        X_train = [fold[0][k][0] for k in range(len(fold[0]))]  # text 
        y_train = [fold[0][k][1] for k in range(len(fold[0]))]  # labels
        
        # Test data
        X_test = [fold[1][k][0] for k in range(len(fold[1]))]  # text 
        y_test = [fold[1][k][1] for k in range(len(fold[1]))]  # labels
        
        # Get sentence embedding by using the FVs.
        X_train_FV = [FV_moVMF(BoWE_doc(wv_model, X_train[k]), vmf_neu) for k in range(len(X_train))]
        X_test_FV = [FV_moVMF(BoWE_doc(wv_model, X_test[k]), vmf_neu) for k in range(len(X_test))]
        
        ## Logistic regression classifier.

        # Use the elements in train_vecs as feature vectors.
        logreg = linear_model.LogisticRegression(C=reg, n_jobs=1, solver='liblinear', multi_class='ovr')
        logreg = logreg.fit(X_train_FV, y_train)

        ## Evaluation.
        acc = evaluate_prediction(logreg, X_test_FV, y_test)
        k_fold_acc.append(acc)
        
    return k_fold_acc


### Fisher vector ###

def FV_GMM(xx, gmm):
    """Create the Fisher vector representation of a document `xx`.
    
    Arguments:
        xx -- document in word-vector representation (e.g. Word2Vec);
              type: array; shape: (T, d).
        gmm -- instance of sklearn mixture.GaussianMixture object.
        
    Returns:
        out -- Fisher vector (derivatives with respect to the means);
               type: array; size: K*d.
               
    Reference:
        S. Clinchant, F. Perronnin, "Aggregating Continuous Word Embeddings 
        for Information Retrieval", ACL (2013).
    """
    
    # Attributes of the GMM.
    means = gmm.means_           # Shape: (K, d)
    covar = gmm.covariances_     # Shape: (K, d)
    weights = gmm.weights_       # Shape: (K, )
    n_comps = gmm.n_components   # Integer scalar
    
    # Encoded document.
    xx = np.atleast_2d(xx)  # Shape: (T, d)    
    T = xx.shape[0]         # Doc. length
    d = xx.shape[1]         # Dimensionality of word/feat. vectors
    
    # Array to store the result.
    out = np.zeros((n_comps, d), dtype=np.float32)  # Shape: (K, d)
    
    # Posterior probabilities.
    probs = gmm.predict_proba(xx)  # Shape: (T, K)
    
    # Soft assignment of a document `xx` to k-th Gaussian. 
    probs_sum = np.sum(probs, 0)[:, np.newaxis]  # Shape: (K, 1)
    
    # Vectorization of the sum over t of `gamma_t(i)*x_t`.
    probs_xx = np.dot(probs.T, xx)  # Shape: (K, d)
    
    # Derivatives with respect to the means.
    d_mean = probs_xx - means * probs_sum  # Shape: (K, d)
    
    # Normalization.
    eps = 1e-6  # Avoids dividing by 0
    np.divide(d_mean, np.sqrt(covar), out=d_mean)
    
    out = d_mean / (np.sqrt(weights.reshape((n_comps, 1)) + eps))
    
    return out.flatten()



def FV_moVMF(xx, vmf):
    """Create the Fisher vector representation of a document `xx`.
    
    Arguments:
        xx -- document in word-vector representation (e.g. Word2Vec);
              type: array; shape: (T, d).
        vmf -- instance of spherecluster VonMisesFisherMixture object.
        
    Returns:
        out -- Fisher vector (derivatives with respect to the mean direction);
               type: array; size: K*d.
               
    References:
        R. Zhang et. al., "Aggregating Neural Word Embeddings for 
        Document Representation", ECIT (2018).
    """
    
    # Attributes of the moVMF.
    #mean_dir = vmf.cluster_centers_  # Shape: (K, d)
    kappa = vmf.concentrations_      # Shape: (K, )
    weights = vmf.weights_           # Shape: (K, )
    n_comps = vmf.n_clusters         # Integer scalar
    
    # Encoded document.
    xx = np.atleast_2d(xx)  # Shape: (T, d)   
    xx = normalize(xx)      # Normalize input data
    T = xx.shape[0]         # Doc. length
    d = xx.shape[1]         # Dimensionality of word/feat. vectors
    
    # Array to store the result.
    out = np.zeros((n_comps, d), dtype=np.float32)  # Shape: (K, d)
    
    # Posterior probabilities.
    probs = vmf.log_likelihood(xx)  # Shape: (T, K)
    
    # Vectorization of the sum over t of `gamma_t(i)*x_t`.
    probs_xx = np.dot(probs, xx)  # Shape: (K, d)
    
    # Derivatives with respect to the mean directions.
    d_mean = d * probs_xx  # Shape: (K, d)
    
    # Normalization.
    eps = 1e-6  # Avoids dividing by 0
    np.divide(d_mean, (kappa.reshape((n_comps, 1)) + eps), out=d_mean)
    
    out = d_mean / (weights.reshape((n_comps, 1)) + eps)
    
    return out.flatten()
