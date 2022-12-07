import numpy as np
import scipy.sparse as sp
import torch
import pickle
import open3d as o3d
import pandas as pd
import math
import random

def read_data():
    fname = "lobster.ply"
    start = 16
    end = 219512
    ct = 0
    ls = []
    for i in open(fname, 'r').readlines():
        if ct >=start and ct<=end:
            x,y,z,_,_,_,_ = i.split()
            ls.append([round(float(x),2),round(float(y),2),round(float(z),2)])
        ct+=1
    ls = np.array(ls)
    pickle.dump(ls, open('lobster.pkl','wb'))

def generate_xyz(ls, name='temp.xyz'):
    df = pd.DataFrame.from_dict({'x':ls[:,0],'y':ls[:,1],'z':ls[:,2]})
    df.to_csv(name, header=False, index=False, sep=' ')

def sample_points(ls, ratio = 0.1):
    tmp = list(ls)
    return np.array(random.sample(tmp, int(len(tmp)*ratio)))
    
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data2():
    dic = pickle.load(open('data2.pkl','rb'))
    features = dic['node_embedding']
    adj = dic['adj_mat']
    pairwise = dic['pairwise']
    features_ = []

    for i in range(len(features)):
        features_.append(torch.FloatTensor(normalize2(features[i])))
        adj[i] = sp.csr_matrix(adj[i])
        adj[i]=  normalize(adj[i]+sp.eye(adj[i].shape[0]))
        adj[i] = sparse_mx_to_torch_sparse_tensor(adj[i])
        pairwise[i] = torch.FloatTensor(pairwise[i])

    return adj, features_, pairwise, features


def load_data3():
    data = pickle.load(open('inference.pkl','rb'))
    features = []
    adj = []
    features_ = []

    for i in range(len(data)):
        features.append(data[i][0])
        features_.append(torch.FloatTensor(normalize2(data[i][0])))
        adj.append(sp.csr_matrix(data[i][1]))
        adj[-1]=  normalize(adj[-1]+sp.eye(adj[-1].shape[0]))
        adj[-1] = sparse_mx_to_torch_sparse_tensor(adj[-1])
        
    return adj, features_, None, features


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize2(x):
    mn = np.min(x)
    mx = np.max(x)
    return (x-mn)/(mx-mn)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def add_to_graph(graph, a, b):
    if a in graph:
        graph[a].add(b)
    else:
        graph[a] = {b}
    
    if b in graph:
        graph[b].add(a)
    else:
        graph[b] = {a}

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def euclidean(x,y):
    return math.sqrt(sum((x-y)**2))

#BFS based graph traversal
def sample_graph(graph, vertices, depth = 5):
    st = random.sample(range(len(graph)), 1)[0]
    q = [st]
    visited = set()
    ct = 0
    while(ct<depth):
        ct+=1
        for i in range(len(q)):
            fr = q.pop(0)
            if fr in visited:
                continue
            visited.add(fr)
            for j in graph[fr]:
                q.append(j)

    sgraph = dict()
    for i in visited:
        sgraph[i] = set()
        for j in graph[i]:
            if j in visited:
                sgraph[i].add(j)

    sedges = {}
    for i in sgraph:
        for j in sgraph[i]:
            sedges[(i,j)] = euclidean(vertices[i], vertices[j])
    
    return sgraph, sedges

def plot_surface(sgraph):
    xyz = []
    for i in sgraph:
        xyz.append(vertices[i])
    xyz = np.array(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    alpha = 3.5
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

def pairwise_distance(sgraph, sedges):
    pairwise = np.ones((len(sgraph), len(sgraph)))
    keys = list(sgraph.keys())
    for i in range(len(sgraph)):
        for j in range(len(sgraph)):
            pairwise[i][j] = float('inf')
            if (keys[i], keys[j]) in sedges:
                pairwise[i][j] = sedges[(keys[i], keys[j])]
    for k in range(len(sgraph)):
        for i in range(len(sgraph)):
            for j in range(len(sgraph)):
                pairwise[i][j] = min(pairwise[i][j], pairwise[i][k]+pairwise[k][j])

    p_max = np.max(pairwise)
    p_min = np.min(pairwise)

    return (pairwise-p_min)/(p_max - p_min)

def process(sgraph, sedges, vertices):
    node_embeddings = []
    adj_mat = []
    for i in sgraph:
        node_embeddings.append(vertices[i])
        adj_mat.append([0]*len(sgraph))
    keys = list(sgraph.keys())
    for i in range(len(sgraph)):
        for j in range(len(sgraph)):
            if (keys[i], keys[j]) in sedges:
                adj_mat[i][j] = 1
        
    return np.array(node_embeddings), np.array(adj_mat)



if __name__=='__main__':
    #MESH GENERATION

    # #read_data()
    # ls = pickle.load(open('lobster.pkl','rb'))
    # #g = generate_graph(ls)
    # s_ls = sample_points(ls, ratio=0.1)
    # generate_xyz(s_ls)
    # pcd = o3d.io.read_point_cloud('temp.xyz')
    # #o3d.visualization.draw_geometries([pcd])
    # alpha = 3.5
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    # #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # o3d.io.write_triangle_mesh("samllerlobster.ply", mesh)
    
    # mesh = o3d.io.read_triangle_mesh('samllerlobster.ply')
    # mesh.compute_vertex_normals()
    # #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # #print(np.asarray(mesh.triangles).shape)
    # #print(np.asarray(mesh.vertices).shape)
    # #print(np.asarray(mesh.triangle_normals).shape)
    # graph = {}
    # vertices = np.asarray(mesh.vertices)
    # normals = np.asarray(mesh.triangle_normals)
    # triangles = np.asarray(mesh.triangles)
    # for i in triangles:
    #     a,b,c = i
    #     add_to_graph(graph, a, b)
    #     add_to_graph(graph, b, c)
    #     add_to_graph(graph, a, c)s
    # pickle.dump([triangles, vertices, normals, graph], open('smallerlobster.pkl','wb'))
    triangles, vertices, normals, graph = pickle.load(open('smallerlobster.pkl','rb'))
    #sgraph, sedges = sample_graph(graph, vertices, depth=5)
    #plot_surface(sgraph)
    #print(pairwise_distance(sgraph, sedges))

    data = {'node_embedding':[], 'adj_mat':[], 'pairwise':[]}
    for i in range(100):
        sgraph, sedges = sample_graph(graph, vertices, depth=5)
        plot_surface(sgraph)
        pairwise = pairwise_distance(sgraph, sedges)
        node_embeddings, adj_mat = process(sgraph, sedges, vertices)
        data['node_embedding'].append(node_embeddings)
        data['adj_mat'].append(adj_mat)
        data['pairwise'].append(pairwise)
        exit(0)

    pickle.dump(data, open('data2.pkl','wb'))