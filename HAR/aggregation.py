import random

import numpy as np
import torch

# import torch
from utils import *
import math

# average
def mean(g_list):
    # print('type', type(g_list))
    # print('size', g_list.shape[0])
    return torch.mean(g_list, dim=0)


# trimmed mean
def TM(g_list, f):
    return g_list.sort(dim=0).values[f:-f].mean(dim=0)


# coordinate-wise median
def median(g_list):
    return g_list.median(dim=0)[0]


# geometric median
def gm(g_list):
    max_iter = 80
    tol = 1e-5
    guess = torch.mean(g_list, dim=0)
    for _ in range(max_iter):
        dist_li = torch.norm(g_list - guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack([w / d for w, d in zip(g_list, dist_li)]), dim=0)
        temp2 = torch.sum(1 / dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= tol:
            break
    return guess


# krum
def krum(g_list, f):
    n = g_list.shape[0]
    # f = round(n * perc_trimming)
    k = n - f - 2
    # print('k', k)
    # x = g_list.permute(0, 2, 1)
    x = g_list
    # print('x shape', type(x), x.shape)
    cdist = torch.cdist(x, x, p=2)
    # print('cdist', cdist, type(cdist))
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    # print('nbhDist', nbhDist, 'nbh', nbh)
    i_star = torch.argmin(nbhDist.sum(1))
    Krum = g_list[i_star, :]

    return Krum


# multi krum
def multi_krum(g_list, f):
    # print('g list size', g_list.shape)
    n = g_list.shape[0]
    k = n - f - 2
    x = g_list
    cdist = torch.cdist(x, x, p=2)
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    print('nbhDist', nbhDist, 'nbh', nbh)
    i_loss, i_star = torch.topk(nbhDist.sum(1), k+1, largest=False)
    print('i star', i_star, 'i oss', i_loss)
    mk_tensor = g_list[i_star,:]
    print('mk tensor', type(mk_tensor), mk_tensor.shape)
    # mkrum = g_dict.mean(1, keepdims = True)
    mkrum = torch.mean(mk_tensor, dim=0)
    assert g_list.shape[0] == mkrum.shape
    return mkrum


# Ref: https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning/blob/main/femnist/release-fedsgd-femnist-bulyan.ipynb
def bulyan(g_list, f):
    n_users = g_list.shape[0]
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = g_list
    all_indices = np.arange(len(g_list))

    while len(bulyan_cluster) < (n_users - 2 * f):
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - f], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - f]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    # print('dim of bulyan cluster ', bulyan_cluster.shape)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    # return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0), np.array(candidate_indices)
    return torch.mean(sorted_params[:n - 2 * f], dim=0)

# # bulyan
# def bulyan(g_list, f):
#   n = len(g_list)
#   d = g_list[0].shape[0]
#   m = n - f - 2
#   cdist = torch.cdist(g_list, g_list, p=2)
#   # Compute all pairwise distances
#   # distances = list([(math.inf, None)] * n for _ in range(n))
#   # for gid_x, gid_y in tools.pairwise(tuple(range(n))):
#   #   dist = gradients[gid_x].sub(gradients[gid_y]).norm().item()
#   #   if not math.isfinite(dist):
#   #     dist = math.inf
#   #   distances[gid_x][gid_y] = (dist, gid_y)
#   #   distances[gid_y][gid_x] = (dist, gid_x)
#   # Compute the scores
#   scores = [None] * n
#   for gid in range(n):
#     dists = cdist[gid]
#     print('dist', type(dists))
#     print('shape', dists.shape)
#     # dists.sort(key=lambda x: x[0])
#     dists.sort(dim=0)
#     dists = dists[:m]
#     scores[gid] = (sum(dist for dist, _ in dists), gid)
#     cdist[gid] = dict(dists)
#   # Selection loop
#   selected = torch.empty(n - 2 * f - 2, d, dtype=g_list[0].dtype, device=g_list[0].device)
#   for i in range(selected.shape[0]):
#     # Update 'm'
#     m = min(m, m - i)
#     # Compute the average of the selected gradients
#     scores.sort(key=lambda x: x[0])
#     selected[i] = sum(g_list[gid] for _, gid in scores[:m]).div_(m)
#     # Remove the gradient from the distances and scores
#     gid_prune = scores[0][1]
#     scores[0] = (math.inf, None)
#     for score, gid in scores[1:]:
#       if gid == gid_prune:
#         scores[gid] = (score - cdist[gid][gid_prune], gid)
#   # Coordinate-wise averaged median
#   m        = selected.shape[0] - 2 * f
#   median   = selected.median(dim=0).values
#   closests = selected.clone().sub_(median).abs_().topk(m, dim=0, largest=False, sorted=False).indices
#   closests.mul_(d).add_(torch.arange(0, d, dtype=closests.dtype, device=closests.device))
#   avgmed   = selected.take(closests).mean(dim=0)
#   # Return resulting gradient
#   return avgmed


# medoid
def medoid(g_list):
    cdist = torch.cdist(g_list, g_list, p=2)
    i_star = torch.argmin(cdist.sum(1))
    print('i star', i_star)
    return g_list[i_star, :]


# soft medoid
def SM(g_list, T):
    n = g_list.shape[0]
    # w = torch.zeros(1, n)
    w_bar = np.zeros((n,))
    cdist = torch.cdist(g_list, g_list, p=2)
    loss = cdist.sum(1)
    for i in range(n):
        w_bar[i] = torch.exp(-loss[i] / T)
    # print('type w', type(w_bar))
    w_sum = np.sum(w_bar)
    if w_sum != 0:
        w = w_bar / w_sum
    g_dict = []
    for i in range(n):
        g_dict.append(g_list[i, :] * w[i])
    t = torch.stack(g_dict)
    return torch.sum(t, 0)


# trimmed soft medoid
def TSM(g_list, f, T):
    # print('g list', g_list.shape)
    n = g_list.shape[0]
    k = n - f
    # w_bar = np.zeros((n,))

    # # Brute force method
    # cdist = torch.cdist(g_list, g_list, p=2)
    # loss = cdist.sum(1)
    # _, nbh = torch.topk(loss, k, largest=False)

    # Trimed method
    l = np.zeros((n,))
    E_cl = np.inf
    for i in range(n):
        d = np.zeros((n,))
        if l[i] < E_cl:
            for j in range(n):
                d[j] = np.linalg.norm(g_list[i] - g_list[j])
            l[i] = (1/n)*np.sum(d)
            for j in range(n):
                l[j] = max(l[j], abs(l[i] - d[j]))
            l_sorted = np.sort(l, )
            E_cl = l_sorted[k]
    Q = np.where(l < E_cl) or np.where(l == E_cl)
    indeces = np.sort(Q[0])
    g_chosen = g_list[indeces,:]
    # g_chosen_list = [g_list[i] for i in indeces]
    # g_chosen = torch.FloatTensor(g_chosen_list)
    # print('g_chosen', g_chosen.shape, 'all', g_list.shape)
    m = g_chosen.shape[0]
    w_bar = np.zeros((m,))
    cdist = torch.cdist(g_chosen, g_chosen, p=2)
    loss = cdist.sum(1)
    for i in range(m):
        w_bar[i] = torch.exp(-loss[i] / T)
    w_sum = np.sum(w_bar)
    if w_sum != 0:
        w = w_bar / w_sum
    g_dict = []
    for i in range(m):
        g_dict.append(g_chosen[i, :] * w[i])
    t = torch.stack(g_dict)
    return torch.sum(t, 0)


    # # calculate the weights
    # for i in range(n):
    #     if i in Q:


    # # TOPRANK argsortlgorithm
    # alpha = 1.0001
    # anchor_size = round(pow(n, 2/3)*pow(math.log(n), 1/3))
    # anchors = random.sample(range(n), anchor_size)
    # print('anchor nodes:', anchors)
    # E_bar = np.zeros((n,))
    # d = np.zeros((anchor_size,n))
    # # print(d[0])
    # # print(d[0][1])
    # # print(d[0,0])
    # for i, anchor in enumerate(anchors):
    #     for j in range(n):
    #         d[i,j] = np.linalg.norm(g_list[anchor] - g_list[j])
    # for j in range(n):
    #     E_bar[j] = (1/anchor_size) * sum(d[:,j])
    # # print('E bar', E_bar)
    # E_sorted = np.sort(E_bar)
    # d_imax = np.amax(d, axis=1)
    # delta_bar = 2 * min(d_imax)
    # # print('d',d, 'd_imax', d_imax, 'del bar', delta_bar/2)
    # print('E sorted k', E_sorted[k], 'hfhg', E_sorted)
    # threshold = E_sorted[k] + 2 * alpha * delta_bar * math.sqrt(math.log(n) / anchor_size)
    # print('E bar', E_bar, 'threshold', threshold)
    # Q = np.where(E_bar < threshold) or np.where(E_bar == threshold)
    # print('brute:', nbh, 'toprank:', Q)

    # # calculate the weights
    # for i in range(n):
    #     if i in Q:
    #         w_bar[i] = torch.exp(-loss[i] / T)
    # w_sum = np.sum(w_bar)
    # if w_sum != 0:
    #     w = w_bar / w_sum
    # # print('weights:', w)
    # g_dict = []
    # for i in range(n):
    #     g_dict.append(g_list[i, :] * w[i])
    # t = torch.stack(g_dict)
    # return torch.sum(t, 0)


# Aggregate the gradients received from the clients
def aggregate_gradients(global_model, active_clients, args, rule):
    message = [[torch.zeros_like(para, requires_grad=False) for para in global_model.parameters()] for _ in
               range(len(active_clients))]
    for id, client in enumerate(active_clients):
        if args.attack and id < args.num_attacked:
            if args.attack_type == 'sign_flipping':
                sigma = -2
                for pi, para in enumerate(client['model'].parameters()):
                    message[id][pi].data.zero_()
                    # message[id][pi].data.add_(1, sigma * para.grad.data)
                    message[id][pi].data.add_(sigma * para.grad.data)
            elif args.attack_type == 'byzantine':
                sigma = 10
                for pi, para in enumerate(client['model'].parameters()):
                    message[id][pi].data.zero_()
                    message[id][pi].data.add_(1, (2 * sigma * torch.rand_like(para.grad.data) - sigma))
        else:
            for pi, para in enumerate(client['model'].parameters()):
                message[id][pi].data.zero_()
                message[id][pi].data.add_(1, para.grad.data)
            # message[client][pi].data.add_(weight_decay, para)
    # print('type', type(message[0][0]), 'size', len(message[0][0]))
    message_f = flatten_list(message, 0)
    # print('type of msg f', type(message_f))
    # trimmed_node_size = round(args.trim_perc * len(active_clients))
    if rule == 'average':
        g_vector = mean(message_f)
    elif rule == 'TM':
        g_vector = TM(message_f, args.trim_size)
    elif rule == 'median':
        g_vector = median(message_f)
    elif rule == 'GM':
        g_vector = gm(message_f)
    elif rule == 'krum':
        g_vector = krum(message_f, args.trim_size)
    elif rule == 'multi_krum':
        g_vector = multi_krum(message_f, args.trim_size)
    elif rule == 'bulyan':
        g_vector = bulyan(message_f, args.trim_size)
    # elif rule == 'multi_bulyan':
    #     g_vector = multi_bulyan(message_f, trimmed_node_size)
    elif rule == 'medoid':
        g_vector = medoid(message_f)
    elif rule == 'SM':
        g_vector = SM(message_f, args.temperature)
    elif rule == 'TSM':
        g_vector = TSM(message_f, args.trim_size, args.temperature)
    else:
        raise Exception("Aggregation rule not defined!")

    g = unflatten_vector(g_vector, global_model)

    for para, grad in zip(global_model.parameters(), g):
        para.data.add_(-args.lr, grad)

    return global_model

