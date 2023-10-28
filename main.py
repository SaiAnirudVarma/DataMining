"""
Author: Jorge A. Niño A01172309
Date: 6 Nov 2020
ITESM Campus QRO
Intelligent Systems
"""
import matplotlib.pyplot as plt
from nltk import word_tokenize, NaiveBayesClassifier
import pandas as pd
import numpy as np
from tweet_miner import TweetMiner
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import time
import sys

def timing(f):
    # Wrapper to time functions
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000))
        return ret
    return wrap


@timing
def train_log_reg_model():
    # Open dataset to train the model
    df_train = pd.read_csv('training.1600000.processed.noemoticon.csv')
    
    # Set the name of the columns
    df_train.columns = ['sentiment', 'id' ,'date', 'query', 'user', 'tweets']

    # Tokenize each item in the tweet column
    word_tokens = [word_tokenize(tweet) for tweet in df_train.tweets]

    # Create an empty list to store the length of the tweets
    len_tokens = []

    # Iterate over the word_tokens list and determine the length of each item
    for i in range(len(word_tokens)):
        len_tokens.append(len(word_tokens[i]))

    # Build the vectorizer
    vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df_train.tweets)

    # Create sparse matrix from the vectorizer
    X = vect.transform(df_train.tweets)

    # Create a DataFrame
    tweets_transformed = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
    tweets_transformed['sentiment'] = df_train['sentiment']
    tweets_transformed['n_words'] = len_tokens

    # Define X and y
    y = tweets_transformed.sentiment
    X = tweets_transformed.drop('sentiment', axis=1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=853)

    # Train a logistic regression model
    log_reg = LogisticRegression(max_iter=100000).fit(X_train, y_train)
    
    # Predict the labels
    y_log_predicted = log_reg.predict(X_test)

    # Print accuracy score and confusion matrix on test set
    print('Accuracy on the logistic regression: ', accuracy_score(y_test, y_log_predicted))

    """
    True negatives: correctly predicted negatives (zeros)
    True positives: correctly predicted positives (ones)
    False negatives: incorrectly predicted negatives (zeros)
    False positives: incorrectly predicted positives (ones)
    """
    print(confusion_matrix(y_test, y_log_predicted)/len(y_test))

    # Save the model to disk
    filename = 'log_reg_model.sav'
    pickle.dump(log_reg, open(filename, 'wb'))

@timing
def train_dec_trees_model():
     # Open dataset to train the model
    df_train = pd.read_csv('training.1600000.processed.noemoticon.csv')
    
    # Set the name of the columns
    df_train.columns = ['sentiment', 'id' ,'date', 'query', 'user', 'tweets']

    # Tokenize each item in the tweet column
    word_tokens = [word_tokenize(tweet) for tweet in df_train.tweets]

    # Create an empty list to store the length of the tweets
    len_tokens = []

    # Iterate over the word_tokens list and determine the length of each item
    for i in range(len(word_tokens)):
        len_tokens.append(len(word_tokens[i]))
    
    """
    TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency. 
    This is very common algorithm to transform text into a meaningful representation 
    of numbers which is used to fit machine algorithm for prediction. 
    TF-IDF is a mesaure of originality of a word by comparing the number of times a word appears in a doc
    with the number of documents the word appears in.
    """

    # Build the vectorizer
    vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df_train.tweets)

    # Create sparse matrix from the vectorizer
    X = vect.transform(df_train.tweets)

    # Create a DataFrame
    tweets_transformed = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
    tweets_transformed['sentiment'] = df_train['sentiment']
    tweets_transformed['n_words'] = len_tokens

    # Define X and y
    y = tweets_transformed.sentiment
    X = tweets_transformed.drop('sentiment', axis=1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=853)

    # Train decision trees with gradient descent model
    dec_trees = DecisionTreeClassifier().fit(X_train, y_train)

    # Predict the labels
    y_predicted = dec_trees.predict(X_test)

    # Print accuracy score and confusion matrix on test set
    print('Accuracy on the decision trees classifier: ', accuracy_score(y_test, y_predicted))

    """
    True negatives: correctly predicted negatives (zeros)
    True positives: correctly predicted positives (ones)
    False negatives: incorrectly predicted negatives (zeros)
    False positives: incorrectly predicted positives (ones)
    """
    print(confusion_matrix(y_test, y_predicted)/len(y_test))

    # Save the model to disk
    filename = 'dec_trees_model.sav'
    pickle.dump(dec_trees, open(filename, 'wb'))

def predict_sentiment(word, method, model, data, x):
    print('Predicting sentiment from tweets using ' + method + "...")
    y_predicted = model.predict(x)
    # Store tweets analyzed in a .csv
    data['sentiment'] = y_predicted
    print("Average sentiment: " + str(pd.to_numeric(data["sentiment"]).mean()))
    data.to_csv(word +'/' + word+'_analyzed.csv')
    print('Analyzed tweets were stored in '+ word +'/' + word+'_analyzed.csv.')


@timing
def main():
    # Ask the user for the data to be searched in twitter and analyzed
    word = input("Enter word to be searched on twitter and analyzed >>> ")

    # Initialize tweet miner
    tweet_miner = TweetMiner(word)

    # Run tweet miner
    tweet_miner.main_function(test=False)

    # Instantiate Pandas Dataframe with the tweets found
    df = tweet_miner.get_df()

    # Tokenize each item in the tweet column
    word_tokens = [word_tokenize(tweet) for tweet in df.text]

    # Create an empty list to store the length of the tweets
    len_tokens = []

    # Iterate over the word_tokens list and determine the length of each item
    for i in range(len(word_tokens)):
        len_tokens.append(len(word_tokens[i]))

    # Build the vectorizer
    vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df.text)

    # Create sparse matrix from the vectorizer
    X = vect.transform(df.text)

    # Create a DataFrame
    tweets_transformed = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
    tweets_transformed['n_words'] = len_tokens

    model_num = int(input('Which model would you like to train and perform sentiment analysis with?\n1. Logistic Regression.\n2. Decision Tree with Gradient Descent.\n3. Both.\n >>> '))
    if model_num == 1:
        if not  os.path.exists('log_reg_model.sav'):
            print('Training logistic regression model...')
            train_log_reg_model()
            print('Logistic regression model trained.')
        # Load the model from disk
        model = pickle.load(open('log_reg_model.sav', 'rb'))
        predict_sentiment(word, 'Logistic Regression', model, df, tweets_transformed)
    elif model_num == 2:
        if not os.path.exists('dec_trees_model.sav'):
            print('Training decision trees model...')
            train_dec_trees_model()
            print('Decision trees model trained.')
        # Load the model from disk
        model = pickle.load(open('dec_trees_model.sav', 'rb'))
        predict_sentiment(word, 'Decision Trees with Gradient Descent', model, df, tweets_transformed)
    elif model_num == 3:
        if not  os.path.exists('log_reg_model.sav'):
            print('Training logistic regression model...')
            train_log_reg_model()
            print('Logistic regression model trained.')
        # Load the model from disk
        model = pickle.load(open('log_reg_model.sav', 'rb'))
        predict_sentiment(word, 'Logistic Regression', model, df, tweets_transformed)
        if not os.path.exists('dec_trees_model.sav'):
            print('Training decision trees model...')
            train_dec_trees_model()
            print('Decision trees model trained.')
        # Load the model from disk
        model = pickle.load(open('dec_trees_model.sav', 'rb'))
        predict_sentiment(word, 'Decision Trees with Gradient Descent', model, df, tweets_transformed)
    else:
        print('Invalid option.')
        sys.exit(0)

main()