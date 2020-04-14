import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
import langid
from sklearn.metrics.pairwise import euclidean_distances
import re



def JS(a, b):
    inter = set(a).intersection(set(b))
    union = list(set(a).union(set(b)))
    return 1- len(inter)/len(union)

def hamming_distance(a,b):
    inter = set(a).intersection(set(b))
    #print(inter)
    return len(inter)

def NLTPprocess(string):
    strings = " ".join([x for x in string.split(" ") if not re.search("http", x)])
    strings = strings.replace("ENGLISH TRANSLATION:", "")
    #strings = re.sub(r'[^\w]', ' ', strings)
    ps = PorterStemmer()
    #print("original string: "+string)
    #print("delete http: "+strings)
    stop_words = set(stopwords.words("english"))
    #print(strings)
    word_tokens = word_tokenize(strings)
    #print(word_tokens)
    filtered_sentence = [ps.stem(w).lower() for w in word_tokens if not w in stop_words and w.encode( 'UTF-8' ).isalpha()]
    #print(filtered_sentence)
    return filtered_sentence

def generateFeature(tweets):
    feature = {}
    n_gram_all = []
    for key in tweets.keys():
        string = tweets[key]
        words = NLTPprocess(string)
        #print(words)
        # output = [word for word in words]
        #output = sorted(set(words))
        output = words
        feature[key] = output
        #print(output)
        n_gram_all.append(output)
    return feature, n_gram_all

#def generateFeatureAll(tweets):



def clustering(people, feature, nk):

    n = len(feature.keys())
    centers = [0]
    i = 0
    phi = np.zeros(n)
    index_set = set([1])

    while i < nk-1:
        max_dist = 0
        for j in range(len(people)):
            x = int(phi[j])
            # print(centers[x][0])

            dist = JS(feature[people[centers[x]]], feature[people[j]])

            # print(dist)
            if dist > max_dist:
                max_dist = dist
                center = j
                index = j + 1
        i += 1
        # print(index)
        index_set.add(index)
        centers.append(center)
        for k in range(len(people)):
            x = int(phi[k])
            dist_x = JS(feature[people[centers[x]]], feature[people[k]])
            dist_i = JS(feature[people[centers[i]]], feature[people[k]])


            if dist_x > dist_i:
                phi[k] = i

    return phi

def clusteringH(people, feature, nk):

    n = len(feature.keys())
    centers = [0]
    i = 0
    phi = np.zeros(n)
    index_set = set([1])

    while i < nk-1:
        max_dist = 0
        for j in range(len(people)):
            x = int(phi[j])

            dist = hamming_distance(feature[people[centers[x]]], feature[people[j]])

            if dist > max_dist:
                max_dist = dist
                center = j
                index = j + 1
        i += 1
        # print(index)
        index_set.add(index)
        centers.append(center)
        for k in range(len(people)):
            x = int(phi[k])

            dist_x = hamming_distance(feature[people[centers[x]]], feature[people[k]])
            dist_i = hamming_distance(feature[people[centers[i]]], feature[people[k]])

            if dist_x > dist_i:
                phi[k] = i

    return phi
