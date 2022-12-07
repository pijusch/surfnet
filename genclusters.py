import pickle
from sklearn.cluster import AgglomerativeClustering
import numpy as np

triangles, vertices, normals, graph = pickle.load(open('smallerlobster.pkl','rb'))
model = AgglomerativeClustering(linkage="ward", n_clusters=100) 
model.fit(vertices)
labels = model.labels_

def pack(idx):
    node_embeddings = vertices[idx]
    adj_mat = []
    for i in range(len(node_embeddings)):
        adj_mat.append([0]*len(node_embeddings))

    for i in range(len(idx)):
        for j in range(len(idx)):
            if idx[i] in graph and idx[j] in graph[idx[i]]:
                adj_mat[i][j] = 1
    
    return node_embeddings, np.array(adj_mat)

data = []

for i in set(labels):
    idx = np.argwhere(labels == i)[:,0]
    data.append(pack(idx))

pickle.dump(data, open('inference.pkl','wb'))