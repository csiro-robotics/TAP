import heapq
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix

import json

def load_data(dataset_source, way, session):
    n1s = []
    n2s = []
    for line in open("../dataset/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_all_nodes = max(max(n1s), max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                        shape=(num_all_nodes, num_all_nodes))

    data_train = sio.loadmat("../dataset/{}_train.mat".format(dataset_source))
    data_test = sio.loadmat("../dataset/{}_test.mat".format(dataset_source))

    labels = np.zeros((num_all_nodes, 1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_all_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj).coalesce()

    num_nodes = []
    for _, v in id_by_class.items():
        num_nodes.append(len(v))

    all_id = [i for i in range(len(num_nodes))]
    base_num = len(all_id) - int(way * session)

    large_res_idex = heapq.nlargest(base_num, enumerate(num_nodes), key=lambda x: x[1])
    
    base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]  # [id_num_tuple[0] for id_num_tuple in large_res_idex], random.sample(all_id, base_num)
    novel_id = list(set(all_id).difference(set(base_id)))

    return adj, features, labels, degree, id_by_class, base_id, novel_id, num_nodes, num_all_nodes

def load_data_corafull(dataset_source, way, session):

    adj, features, labels, node_names, attr_names, class_names, metadata=load_npz_to_sparse_graph('../dataset/{}.npz'.format(dataset_source))
    class_list_train,class_list_valid,class_list_test=json.load(open('../dataset/{}_class_split.json'.format(dataset_source)))
    sparse_mx = adj.tocoo().astype(np.float32)
    indices =np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    
    n1s=indices[0].tolist()
    n2s=indices[1].tolist()

    num_all_nodes = max(max(n1s), max(n2s)) + 1

    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)
    
    adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
    adj= sparse_mx_to_torch_sparse_tensor(adj).coalesce()
    features=features.todense()
    features = torch.FloatTensor(features)
    labels=torch.LongTensor(labels).squeeze()
            
        
    class_list =  class_list_train+class_list_valid+class_list_test

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels.numpy().tolist()):
        id_by_class[cla].append(id)


    num_nodes = []
    for _, v in id_by_class.items():
        num_nodes.append(len(v))

    all_id = [i for i in range(len(num_nodes))]    
    base_num = len(all_id) - int(way * session)

    large_res_idex = heapq.nlargest(base_num, enumerate(num_nodes), key=lambda x: x[1])
    
    base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]  # [id_num_tuple[0] for id_num_tuple in large_res_idex], random.sample(all_id, base_num)
    novel_id = list(set(all_id).difference(set(base_id)))
    print(base_id)
    print(novel_id)

    return adj, features, labels, degree, id_by_class, base_id, novel_id, num_nodes, num_all_nodes

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata

def task_generator(id_by_class, n_way, k_shot, novel_cid, seed=12345):
    # split base, novel and seen classes, not just base and novel classes
    random.seed(seed)
    novel_class_selected = random.sample(novel_cid, n_way)
    novel_id_support = []
    novel_id_query = []
    for cla in novel_class_selected:
        temp = random.sample(id_by_class[cla], len(id_by_class[cla]))
        novel_id_support.extend(temp[:k_shot])
        novel_id_query.extend(temp[k_shot:])
    return np.array(novel_id_query), np.array(novel_id_support), novel_class_selected

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adj_tensor(sp_adj, a_size):
    # sp_adj = sp_adj + torch.sparse_coo_tensor(torch.eye(a_size), device=sp_adj.device)
    rowsum = sp_adj.sum(1)
    d_inv_sqrt = rowsum.pow_(-0.5)
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm = torch.matmul(torch.matmul(d_mat_inv_sqrt, sp_adj), d_mat_inv_sqrt)

    return norm

def get_base_adj(adj, base_id, labels):
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)

    mask = []
    for i in range(I.shape[1]):
        if labels[I[0, i]] in base_id and labels[I[1, i]] in base_id:
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)

    I_base = I[:, mask]
    V_base = V[mask]

    base_adj = torch.sparse_coo_tensor(I_base, V_base, (dim_base, dim_base)).coalesce()

    return base_adj

def get_incremental_adj(adj, base_id, novel_id_support, novel_id_query, labels):
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)
    novel_idx = np.append(novel_id_support, novel_id_query)

    mask = []
    for i in range(I.shape[1]):
        if (labels[I[0, i]] in base_id and labels[I[1, i]] in base_id) or \
                (I[0, i] in novel_idx and I[1, i] in novel_idx):
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)
    I_incremental = I[:, mask]
    V_incremental = V[mask]

    incremental_adj = torch.sparse_coo_tensor(I_incremental, V_incremental, (dim_base, dim_base)).coalesce()

    return incremental_adj

## get incremental adj with base and novel classes
def get_incremental_adj_novel(adj, base_class_id, novel_class_id, labels):
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)
    # novel_idx = np.append(novel_id_support, novel_id_query)
    mask = []
    for i in range(I.shape[1]):
        if (labels[I[0, i]] in base_class_id and labels[I[1, i]] in base_class_id) or \
                (labels[I[0, i]] in novel_class_id and labels[I[1, i]] in novel_class_id):
        # if I[0, i] in novel_idx and I[1, i] in novel_idx:
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)
    I_incremental = I[:, mask]
    V_incremental = V[mask]

    incremental_adj = torch.sparse_coo_tensor(I_incremental, V_incremental, (dim_base, dim_base)).coalesce()

    return incremental_adj

def get_base_adj_mask(adj, base_class_id, edge_labels):
    I = adj.indices()
    V = adj.values()

    mask_b_0 = torch.zeros(V.shape)
    mask_b_1 = torch.zeros(V.shape)
    for i in base_class_id:
        mask_b_0[edge_labels[0]==i] = 1.
        mask_b_1[edge_labels[1]==i] = 1.
    mask_base = mask_b_0 * mask_b_1

    mask_base = mask_base.type(torch.bool).to(adj.device)

    return I[:, mask_base]

def split_adj_with_class_groups(adj, base_class_id, labels, edge_labels, n_way, noise_rate=0.3):
    I = adj.indices()
    V = torch.ones(I.size(1), device=I.device) #adj.values()
    dim_adj = len(labels)

    base_class_id = random.sample(base_class_id, len(base_class_id))
    class_group_list = []
    for j in range(int(len(base_class_id)/n_way)):
        class_group_list.append(base_class_id[n_way*j:n_way*(j+1)])

    mask_adj = torch.zeros(V.shape)
    for k in range(len(class_group_list)):
        mask_n_0 = torch.zeros(V.shape)
        mask_n_1 = torch.zeros(V.shape)
        for i in class_group_list[k]:
            mask_n_0[edge_labels[0]==i] = 1.
            mask_n_1[edge_labels[1]==i] = 1.
        mask_temp = mask_n_0 * mask_n_1
        mask_adj = mask_adj + mask_temp

    mask_adj = mask_adj.type(torch.bool)
    I_incremental = I[:, mask_adj]

    if noise_rate:
        add_edges = generate_random_edge_list(I_incremental, p=noise_rate)
        I_incremental = torch.cat((I_incremental, add_edges), dim=1)

    return I_incremental

def update_incremental_adj(adj, I_old, novel_class_id, labels, edge_labels, p=0.3, noise=True):
    # keep fix base_adj and seen_adj, combine with novel_adj

    I = adj.indices()
    V = adj.values()

    mask_n_0 = torch.zeros(V.shape)
    mask_n_1 = torch.zeros(V.shape)
    for i in novel_class_id:
        mask_n_0[edge_labels[0] == i] = 1.
        mask_n_1[edge_labels[1] == i] = 1.
    mask_novel = mask_n_0 * mask_n_1

    mask_novel = mask_novel.type(torch.bool)
    I_novel = I[:, mask_novel]
    V_novel = V[mask_novel]
    if noise:
        mask_novel = mask_novel.type(torch.bool)
        # edge_index_noise, add_edges = add_random_edge(I[:, mask_novel], p=0.3)
        add_edges = generate_random_edge_list(I[:, mask_novel], p)
        I_novel = torch.cat((I_novel, add_edges), dim=1)
        V_noise = torch.ones(add_edges.size(1), dtype=torch.float, device=V_novel.device)
        V_novel = torch.cat((V_novel, V_noise))

    I_incremental = torch.cat((I_old, I_novel), dim=1)

    return I_incremental

def generate_random_edge_list(edge_list, p=0.3):
    add_edge_list = torch.zeros((2, int(edge_list.size(1)*p)), device=edge_list.device, dtype=torch.int)
    add_edge_list[0] = edge_list[0][torch.randperm(add_edge_list.size(1))]
    add_edge_list[1] = edge_list[1][torch.randperm(add_edge_list.size(1))]
    return add_edge_list

def adjust_labels(labels, selected_classes, n_class):
    # labels --> one-hot 
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=n_class)
    # adjust labels according to selected class index
    one_hot_labels = one_hot_labels[:, selected_classes]
    # one-hot --> labels
    labels_digit = torch.argmax(one_hot_labels, dim=1)

    return labels_digit


def edge_label_mapping(adj, labels):
    I = adj.coalesce().indices()
    wdge_label = torch.zeros(I.shape)
    wdge_label[0, :] = labels[I[0, :]]
    wdge_label[1, :] = labels[I[1, :]]

    return wdge_label


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def base_train_val_split(id_by_class, base_class_id, seed=123):
    # sample class indices
    random.seed(seed)
    base_id_train = []
    base_id_val = []

    for cla in base_class_id:
        num_train = int(0.8*len(id_by_class[cla]))
        temp = random.sample(id_by_class[cla], len(id_by_class[cla]))
        base_id_train.extend(temp[:num_train])
        base_id_val.extend(temp[num_train:])

    return base_id_train, base_id_val
