
import pandas as pd
import re
from tools import *
import numpy as np
from collections import Counter
import langid

import inputData

#tweets = inputData.allUser()
#tweets = inputData.actualTweeters()
tweets = inputData.actualTweetersByUser()
feature, n_gram_all = generateFeature(tweets)
print(n_gram_all)

print(len(n_gram_all))

# hash vectorizer instance
from sklearn.feature_extraction.text import HashingVectorizer
hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)

# features matrix X
X = hvec.fit_transform(n_gram_all)
print(X.shape)
print(X)

#------imensionality Reduction with t-SNE and shown in 2d dimensions------

from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, perplexity=5)
X_embedded = tsne.fit_transform(X)
#print(X_embedded)


from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", 1)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette,  s = 60)

plt.title("ISIS related tweeters in 2D dimensions")
# plt.savefig("plots/t-sne_covid19.png")
plt.show()

#---------K-means clustering----------

from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
y_pred = kmeans.fit_predict(X)

# sns settings

sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette, s = 60)
plt.title("ISIS related tweets - Kmeans Clustering")
# plt.savefig("plots/t-sne_covid19_label.png")
plt.show()
print(y_pred)



#------AgglomerativeClustering--------


from sklearn.cluster import AgglomerativeClustering
agglomer = AgglomerativeClustering(n_clusters=k)
y_pred = agglomer.fit(X.todense()).labels_
print(y_pred)

sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette, s = 60)
plt.title("ISIS related tweets - AgglomerativeClustering")
# plt.savefig("plots/t-sne_covid19_label.png")
plt.show()

