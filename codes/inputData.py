import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import networkx as nx
import pandas as pd
import re
from tools import *
import numpy as np
from collections import Counter
import langid
from sklearn.feature_extraction.text import HashingVectorizer


def allUser():

    dataset = pd.read_csv("tweets.csv")

    #people = set([user for user in dataset["username"]])
    tweets = {}
    followers = {}
    for user, tweet in zip(dataset["username"], dataset["tweets"]):

        if langid.classify(tweet)[0] == "en":
            #print(tweet)
            if user in tweets:
                tweets[user] += tweet
            else:
                tweets[user] = tweet
                followers[user] = dataset[dataset['username'] == user]["followers"]
    return tweets

def actualTweeters():
    dataset = pd.read_csv("tweets.csv")

    retweets = []
    actual_tweets = []
    for user, tweet in zip(dataset['username'], dataset['tweets']):
        match = re.search(r'^\bRT\b', tweet)
        if match == None:
            actual_tweets.append([user, tweet])
        else:
            retweets.append([user, tweet])

    actual_tweets = np.array(actual_tweets)
    #retweets = np.array(retweets)

    tweets = {}
    followers = {}
    p = []
    i = 0
    for line in actual_tweets:
        user = line[0]
        tweet = line[1]
        if langid.classify(tweet)[0] == "en":
            # print(tweet)
            key = user + str(i)
            tweets[key] = tweet
            i += 1
            '''
            if user in tweets:
                tweets[user] += tweet
            else:
                tweets[user] = tweet
                p.append(user)
                followers[user] = dataset[dataset['username'] == user]["followers"]
                
            '''
    return tweets


def actualTweetersByUser():
    dataset = pd.read_csv("tweets.csv")

    retweets = []
    actual_tweets = []
    for user, tweet in zip(dataset['username'], dataset['tweets']):
        match = re.search(r'^\bRT\b', tweet)
        if match == None:
            actual_tweets.append([user, tweet])
        else:
            retweets.append([user, tweet])

    actual_tweets = np.array(actual_tweets)
    # retweets = np.array(retweets)

    tweets = {}
    followers = {}
    p = []
    i = 0
    for line in actual_tweets:
        user = line[0]
        tweet = line[1]
        if langid.classify(tweet)[0] == "en":
            # print(tweet)
            #key = user + str(i)
            #tweets[key] = tweet
            #i += 1

            if user in tweets:
                tweets[user] += tweet
            else:
                tweets[user] = tweet
                p.append(user)
                followers[user] = dataset[dataset['username'] == user]["followers"]
    return tweets


