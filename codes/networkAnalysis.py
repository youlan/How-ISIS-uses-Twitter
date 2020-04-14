import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import re
import numpy as np
from collections import Counter
import langid
from tools import *

plt.style.use('ggplot')

dataset = pd.read_csv("tweets.csv")


retweets = []
actual_tweets = []
for user, tweet in zip(dataset['username'], dataset['tweets']):
    match = re.search(r'^\bRT\b', tweet)
    if match == None:
        actual_tweets.append([user,tweet])
    else:
        retweets.append([user,tweet])

actual_tweets = np.array(actual_tweets)
retweets = np.array(retweets)

in_set = []
not_in_set = []
for record in actual_tweets:
    #print(record)
    match = re.findall(r'@\w*', record[1])
    if match != []:
        for name in match:
            if (name[1:] in dataset['username'].unique()) and (record[0] != name[1:]):
                in_set.append([record[0], name[1:]])
            elif record[0] != name[1:]:
                not_in_set.append([record[0], name[1:]])

in_set = np.array(in_set)

not_in_set = np.array(not_in_set)
#print(in_set.shape)
sender_count = Counter(in_set[:,0])
receiver_count = Counter(in_set[:,1])
top_5_senders = sender_count.most_common(5)
top_5_receivers = receiver_count.most_common(5)


graph = nx.Graph()
all_users = list(set(in_set[:,0]) | set(in_set[:,1]))
print(len(all_users))
graph.add_nodes_from(all_users, count=10)
node_colours = []


people = set(actual_tweets[:,0])
tweets = {}
followers = {}
p = []
for line in actual_tweets:
    user = line[0]
    tweet = line[1]
    if langid.classify(tweet)[0] == "en":
        #print(tweet)
        if user in tweets:
            tweets[user] += tweet
        elif user in all_users:
            tweets[user] = tweet
            p.append(user)
            followers[user] = dataset[dataset['username'] == user]["followers"]

feature, n_grams = generateFeature(tweets)



for node in graph.nodes():
    if node in (set(in_set[:, 0]) & set(in_set[:, 1])):
        node_colours.append('g')
    elif node in np.unique(in_set[:,0]):
        node_colours.append('r')
    elif node in np.unique(in_set[:,1]):
        node_colours.append('b')

edges = {}
occurrence_count = Counter(map(tuple, in_set))
for (sender, receiver), count in occurrence_count.items():
    if (receiver, sender) in edges.keys():
        edges[(receiver, sender)] = edges[(receiver, sender)] + count
    else:
        edges[(sender, receiver)] = count

for (sender, receiver), count in edges.items():
    graph.add_edge(sender, receiver, weight=count)

followers = {}
tweet_num = {}
for username in all_users:
    followers[username] = dataset[dataset['username'] == username]['followers'].unique()[-1]
    tweet_num[username] = dataset[dataset['username'] == username]['tweets'].count()

sizes = [(followers[n] / tweet_num[n]) * 50 for n in graph.nodes()]
weights = []

#print(graph.edges[('RamiAlLolah', 'MaghrabiArabi')])
for (u, v) in graph.edges():
    #print(graph.edges[('RamiAlLolah', 'MaghrabiArabi')])
    weights.append(graph.edges[(u,v)]["weight"]/2)
#print(len(weights))
#weights = [graph.edges[u][v]['weight'] / 2 for (u, v) in graph.edges()]


#-------Color nodes with K-means clustering, distance with JS similarity--------

k = 10

phi = clustering(p, feature, k)

plt.figure(figsize=(12, 12))
nx.draw(graph, pos=nx.spring_layout(graph, iterations=10),
        node_color=phi, with_labels=True, width=weights)
plt.title("ISIS users network colored with K-means clustering based on JS distance")
plt.show()

#-------Color nodes with K-means clustering, distance with Hamming distance--------


phi = clusteringH(p, feature, k)

plt.figure(figsize=(12, 12))
nx.draw(graph, pos=nx.spring_layout(graph, iterations=10),
        node_color=phi, with_labels=True, width=weights)
plt.title("ISIS users network colored with K-means clustering based on Hamming distance")
plt.show()
