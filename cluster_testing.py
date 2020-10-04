import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def get_keys_vectors(words):
    """
    returns keys, vectors after applying w2v to a list of words
    """
    model = Word2Vec.load('Snippext_public/word2vec/rest_w2v.model')
    w2v = model.wv
    keys = [word.lower() for word in words if word in w2v.vocab]
    vectors = [w2v.word_vec(word) for word in keys]
    return keys, vectors
   
def print_clusters(words, clusters):
    """
    prints the clusters one-by-one
    """
    n_clusters = max(clusters)
    start_cluster = min(clusters)
    for i in range(start_cluster, n_clusters + 1):
        words_in_cluster = np.array(words)[np.array(clusters) == i]
        print(f'Cluster {i}: {words_in_cluster}')
        
def find_lower_dim(vectors, threshold = 0.7):
    """
    returns # of dimensions (n) necessary for  sum of first n 
    """
    pca = PCA()
    pca.fit(np.array(vectors))
    sv = pca.singular_values_
    total = sum(sv**2)
    pcts = [sv[0]**2/total]
    for i in range(1, len(sv)):
        pcts.append(pcts[i - 1] + sv[i]**2/total)
    new_dim = len(np.array(pcts)[np.array(pcts) < .7])
    return new_dim + 1

def reduce_dim(vectors ,new_dim):
    pca = PCA(n_components= new_dim)
    return pca.fit_transform(vectors)

def tsne_proj(vectors):
    tsne = TSNE()
    return tsne.fit_transform(vectors)

def view_words(keys , vectors):
    """
    pass in your words and their respresentations to view them
    """
    tsne_vectors = tsne_proj(vectors)
    fig = plt.figure(figsize= (25, 19))
    plt.scatter(tsne_vectors[:,0], tsne_vectors[:,1], s = 50)
    for i in range(len(tsne_vectors)):
        plt.annotate(keys[i], tsne_vectors[i], size = 20)    
    plt.show()
    
def radius_cluster(epsilon, word_vectors):
    """
    word_vectors must be ordered with vectors of most common words coming first
    """
    cluster = 1
    ret = [0]*(len(word_vectors))
    while 0 in ret:
        to_cluster_index = ret.index(0)
        to_cluster_word = word_vectors[to_cluster_index]
        distances = [np.linalg.norm(x - to_cluster_word) for x in word_vectors]
        indices = []
        for i in range(len(distances)):
            if distances[i] < epsilon:
                indices.append(i)
        for idx in indices:
            ret[idx] = cluster
        cluster += 1
    return ret

    


    
