from __future__ import division
from __future__ import print_function

import copy
import time
import argparse
import numpy as np
import os
import os.path as osp
from copy import deepcopy
import torch
import torch.optim as optim
from criterion import *
from data_split import *
from models import *
from datetime import datetime
from torch_geometric.utils import k_hop_subgraph
from kmeans_refine import *
from trainer import *
from configuration import *


if __name__ == '__main__':
    t_total = time.time()
    args = update_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")
    set_randomseed(args.seed)

    print('*******************************')
    print('Loading dataset...' + args.dataset)
    if args.dataset == 'cora_full':
        adj, features, labels, degrees, id_by_class, base_class_id, novel_class_id, num_nodes, num_all_nodes = load_data_corafull(
            args.dataset, args.way, args.session)
    else:
        adj, features, labels, degrees, id_by_class, base_class_id, novel_class_id, num_nodes, num_all_nodes = load_data(
            args.dataset, args.way, args.session)

    edge_labels = edge_label_mapping(adj, labels)
    num_classes = (labels.max() + 1).item()
    base_trainset, base_valset = base_train_val_split(id_by_class, base_class_id)
    num_base_nodes = len(base_trainset) + len(base_valset)

    print('Few-shot setting:')
    print('Few-shot setting:' + str(args.way) + '-way \t' + str(args.shot) + '-shot')
    print('Total classes:{}, base classes:{}, novel classes:{}, incremental sessions:{}'.format(len(id_by_class),
                                                                                         len(base_class_id),
                                                                                         len(novel_class_id),
                                                                                         args.session))
    print('*******************************')

    encoder = GAT_Encoder(nfeat=features.shape[1],
                          nhid=args.hidden,
                          dropout=args.dropout,
                          n_head=args.gat_heads)
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    encoder.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    incre_base_class_selected = deepcopy(base_class_id)
    novel_class_left = deepcopy(novel_class_id)
    angular_criterion = AngularPenaltySMLoss(loss_type=args.loss_type, s=args.tau, m=args.k)
    adj_base_indices = get_base_adj_mask(adj.coalesce(), base_class_id, edge_labels)

    labels_train = adjust_labels(labels[base_trainset], base_class_id, num_classes)
    labels_val = adjust_labels(labels[base_valset], base_class_id, num_classes)

    n_mix_class = 2 * len(base_class_id)
    labels_groups = (labels_train + len(base_class_id)).to(device)
    list_train_acc = []

    if args.pertb and not args.eye_pertb:
        n_classifier_out = n_mix_class
        labels_eye = torch.LongTensor([i + len(base_class_id) for i in labels_train.cpu()]).to(device)
    elif args.pertb and args.eye_pertb:
        n_classifier_out = n_mix_class + len(base_class_id)
        labels_eye = torch.LongTensor([i + n_mix_class for i in labels_train.cpu()]).to(device)
    elif not args.pertb and args.eye_pertb:
        n_classifier_out = 2 * len(base_class_id)
        labels_eye = torch.LongTensor([i + len(base_class_id) for i in labels_train.cpu()]).to(device)
    else:
        n_classifier_out = len(base_class_id)
        labels_eye = copy.deepcopy(labels_train).cuda()

    base_classifier = BaseClassifier(args.hidden, n_classifier_out)
    optimizer_classifier = optim.Adam(base_classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    labels_train = labels_train.to(device)
    labels_val = labels_val.to(device)
    base_classifier = base_classifier.to(device)

    best_encoder_dict = deepcopy(encoder.state_dict())
    best_classifier_dict = deepcopy(base_classifier.state_dict())

    for episode in range(args.episodes):
        class_group_adj = split_adj_with_class_groups(adj, base_class_id, labels, edge_labels, args.way)

        acc_train, loss_train, acc_val, eval_loss = train_mixup(encoder, features, adj_base_indices,
                                                                base_trainset,
                                                                base_valset, labels_train, labels_val,
                                                                base_classifier,
                                                                class_group_adj, labels_groups,
                                                                labels_eye, angular_criterion,
                                                                optimizer_encoder, optimizer_classifier, args)
        print("episode: {:4d}, train_acc: {:.4f}, train_loss: {:.4f} | eval_acc: {:.4f}, eval_loss: {:.4f}".
              format(episode, acc_train, loss_train, acc_val, eval_loss))
        list_train_acc.append(acc_train)


    # Incremental finetune and test
    seen_class_selected = []
    seen_id_support = []
    seen_id_query = []
    incre_base_class_selected = deepcopy(base_class_id)
    novel_class_left = deepcopy(novel_class_id)

    base_class_selected = base_class_id
    base_id_query = base_valset + base_trainset
    cls_wise_prototypes = get_base_prototypes(encoder, features, adj_base_indices,
                                              labels, base_trainset, base_class_selected)
    base_acc = test_base(encoder, features, adj_base_indices, base_valset + base_trainset, cls_wise_prototypes,
                         labels, base_class_selected, num_classes, device)

    stored_cls_wise_prototypes = copy.deepcopy(cls_wise_prototypes.cpu().detach())

    n_novel_list = [0]
    novel_encoder = GAT_Encoder(nfeat=features.shape[1],
                                nhid=args.hidden,
                                dropout=args.dropout,
                                n_head=args.gat_heads)
    novel_encoder.to(device)
    novel_encoder.load_state_dict(copy.deepcopy(encoder.state_dict()))

    optimizer_novel_encoder = optim.Adam(filter(lambda p: p.requires_grad, novel_encoder.parameters()),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

    for idx in range(args.session):
        if idx == 0:
            adj_old_indices = copy.deepcopy(adj_base_indices)
        else:
            adj_old_indices = incremental_adj_indices

        novel_id_query, novel_id_support, novel_class_selected = \
            task_generator(id_by_class, args.way, args.shot, novel_class_left, seed=args.seed)

        print("novel_class_selected:{}".format(novel_class_selected))

        incremental_adj_indices = update_incremental_adj(adj.coalesce(), adj_old_indices,
                                                           novel_class_selected, labels,
                                                           edge_labels, noise=True)
        all_encouter_class = base_class_selected + seen_class_selected + novel_class_selected
        base_seen_classes = base_class_selected + seen_class_selected

        # # finetune
        if args.finetune:
            labels_novel_support_finetune = (
                    adjust_labels(labels[novel_id_support], novel_class_selected, num_classes) + len(
                base_seen_classes)).to(device)
            labels_novel_query_finetune = (
                    adjust_labels(labels[novel_id_query], novel_class_selected, num_classes) + len(
                base_seen_classes)).to(device)

            for eps in range(args.episodes_ft):
                finetune(novel_encoder, features, incremental_adj_indices,
                         novel_id_support,novel_id_query,
                         labels_novel_support_finetune,
                         labels_novel_query_finetune,
                         cls_wise_prototypes, eps,
                         novel_class_selected, optimizer_novel_encoder, angular_criterion, args)

            # moving average
            update_momentum_encoder_b_to_n(encoder, novel_encoder, beta=args.beta)

            # # prototype shift
            adjust_cls_prototypes = prototype_shift(encoder,novel_encoder, features,
                                                    incremental_adj_indices, cls_wise_prototypes,
                                                    novel_id_support, args.sigma)

            cls_wise_prototypes = adjust_cls_prototypes

        novel_cls_prototypes = update_novel_prototypes(novel_encoder, features,incremental_adj_indices,
                                                       novel_id_support, cls_wise_prototypes,
                                                       novel_class_selected, novel_id_query, args)

        cls_wise_prototypes = torch.cat((cls_wise_prototypes, novel_cls_prototypes), dim=0)

        n_novel_list.append(n_novel_list[-1] + len(novel_id_query))

        all_acc_test, base_acc_test, novel_acc_test \
            = test_incremental(novel_encoder, features, incremental_adj_indices, base_id_query, novel_id_query, seen_id_query,
                           base_class_selected, novel_class_selected, seen_class_selected, cls_wise_prototypes,
                           n_novel_list, labels, num_classes, device)

        if args.finetune:
            update_momentum_encoder_n_to_b(encoder, novel_encoder)

        seen_class_selected.extend(novel_class_selected)
        seen_id_support.extend(novel_id_support)
        seen_id_query.extend(novel_id_query)

        print("Session: {:2d}, all_test_acc: {:.4f}, "
              "base_test_acc: {:.4f}, "
              "novel_test_acc: {:.4f}".format(idx + 1, all_acc_test, base_acc_test, novel_acc_test))

        incre_base_class_selected.extend(novel_class_selected)
        novel_class_left = list(set(novel_class_left) - set(novel_class_selected))



