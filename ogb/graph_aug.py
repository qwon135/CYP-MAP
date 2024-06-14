import torch
import numpy as np

def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    # token = data.x.mean(dim=0)
    token = [118, 3, 11, 6, 9, 5, 5, 0, 0]
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.long)

    return data

def mask_edges(data, aug_ratio):

    edge_num, feat_dim = data.edge_attr.size()
    mask_num = int(edge_num * aug_ratio)

    # token = data.x.mean(dim=0)
    token = [4, 5, 0]
    idx_mask = np.random.choice(edge_num, mask_num, replace=False)
    data.edge_attr[idx_mask] = torch.tensor(token, dtype=torch.long)

    return data

def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_delete]
    data.ring_index = data.ring_index[:, idx_delete]
    data.edge_attr = data.edge_attr[idx_delete]

    return data
def drop_nodes(data, aug_ratio):    
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    _, ring_num = data.ring_index.size()

    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    ring_index = data.ring_index.numpy()

    edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])
    edge_index_index = [n for n in range(edge_num) if (edge_index[0, n] not in idx_drop) and (edge_index[1, n] not in idx_drop)]

    data.edge_index = torch.from_numpy(edge_index.T[edge_index_index].T)
    data.ring_index = torch.from_numpy(ring_index.T[edge_index_index].T)

    data.x = data.x[idx_nondrop]
    data.edge_attr = data.edge_attr[edge_mask]

    return data

def subgraph(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
    edge_mask = np.array([n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

    edge_index = data.edge_index.numpy()
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data