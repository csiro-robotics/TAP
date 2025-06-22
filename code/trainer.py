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

def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_prototypes(encoder, features, curr_adj, labels, base_trainset, class_selected, device=torch.device('cuda')):
    encoder.eval()
    embeddings, _ = encoder(features, curr_adj)
    embeddings = embeddings[base_trainset]

    cls_wise_prototypes = torch.zeros((len(class_selected), embeddings.size(1)))
    labels_tr = labels[base_trainset]

    for cla in range(len(class_selected)):
        mask = labels_tr.eq(class_selected[cla])
        cls_wise_prototypes[cla] = embeddings[mask].mean(0)
    cls_wise_prototypes = cls_wise_prototypes.to(device)

    return cls_wise_prototypes

def train_mixup(encoder, features, curr_adj, base_train_set, base_val_set, labels_train, labels_val, classifier,
                group_adj, labels_group, eye_labels, criterion, optimizer_encoder, optimizer_classifier, args):
    encoder.train()
    classifier.train()
    optimizer_encoder.zero_grad()
    optimizer_classifier.zero_grad()

    embeddings, _ = encoder(features, curr_adj)
    train_logits = classifier(embeddings, args.loss_type)[base_train_set]

    # eye augmentation
    if args.eye_pertb:
        eye_adj = torch.eye(curr_adj.size(0), dtype=torch.float32, device=curr_adj.device).to_sparse()
        embeddings_eye, _ = encoder(features, eye_adj.indices())
        embeddings_eye = embeddings_eye[base_train_set]
        train_logits_eye = classifier(embeddings_eye, args.loss_type)
    # mixup augmentation
    if args.pertb:
        embeddings_mix, _ = encoder(features, group_adj)  # group_adj.indices()
        train_logits_mix = classifier(embeddings_mix, args.loss_type)[base_train_set]

    if args.pertb and args.eye_pertb:
        angular_loss_1 = criterion(train_logits, labels_train)
        angular_loss_2 = criterion(train_logits_mix, labels_group)
        angular_loss_3 = criterion(train_logits_eye, eye_labels)
        angular_loss = args.alpha * angular_loss_1 + (1. - args.alpha) * angular_loss_2 * 0.5 + (
                1. - args.alpha) * angular_loss_3 * 0.5
    elif args.pertb and not args.eye_pertb:
        angular_loss_1 = criterion(train_logits, labels_train)
        angular_loss_2 = criterion(train_logits_mix, labels_group)
        angular_loss = args.alpha * angular_loss_1 + (1. - args.alpha) * angular_loss_2
    elif not args.pertb and args.eye_pertb:
        angular_loss_1 = criterion(train_logits, labels_train)
        angular_loss_3 = criterion(train_logits_eye, eye_labels)
        angular_loss = args.alpha * angular_loss_1 + (1. - args.alpha) * angular_loss_3
    else:
        angular_loss = criterion(train_logits, labels_train)

    loss_train = angular_loss
    loss_train.backward()
    optimizer_encoder.step()
    optimizer_classifier.step()

    train_output = train_logits.cpu().detach()
    labels_train = labels_train.cpu().detach()
    acc_train = accuracy(train_output, labels_train)

    with torch.no_grad():
        encoder.eval()
        classifier.eval()
        embeddings, _ = encoder(features, curr_adj)
        val_logits = classifier(embeddings, args.loss_type)[base_val_set]
        eval_loss = criterion(val_logits, labels_val)
        val_output = val_logits.cpu().detach()
        labels_val = labels_val.cpu().detach()
        acc_val = accuracy(val_output, labels_val)
    return acc_train, loss_train, acc_val, eval_loss

def test_base(encoder, features, curr_adj, base_val_set, cls_prototypes, labels,
              base_class_selected, num_classes, device=torch.device('cuda')):
    encoder.eval()
    embeddings, _ = encoder(features, curr_adj)
    val_embeddings = embeddings[base_val_set]

    cls_prototypes = F.normalize(cls_prototypes, p=2, dim=-1)
    val_embeddings = F.normalize(val_embeddings, p=2, dim=-1)

    val_logits = F.linear(val_embeddings, cls_prototypes)

    labels_val = (adjust_labels(labels[base_val_set], base_class_selected, num_classes)).to(device)

    val_output = val_logits.cpu().detach()
    labels_val = labels_val.cpu().detach()
    acc_val = accuracy(val_output, labels_val)
    print('Session 0: all_test acc:{:.4f}, base_test_acc: {:.4f}, novel_test_acc: {:.4f}'.format(
        acc_val, acc_val, 0.))

    return acc_val

def test_incremental(novel_encoder, features, curr_adj, base_id_query, novel_id_query, seen_id_query,
                 base_class_selected, novel_class_selected, seen_class_selected, cls_wise_feature_prototype,
                 n_novel_list, labels, num_classes, device=torch.device('cuda')):
    novel_encoder.eval()

    all_id_query = list(base_id_query) + list(seen_id_query) + list(novel_id_query)
    all_encounter_class = base_class_selected + seen_class_selected + novel_class_selected

    proto_list = F.normalize(cls_wise_feature_prototype, p=2, dim=-1)
    embeddings, _ = novel_encoder(features, curr_adj)

    novel_query_embeddings = embeddings[novel_id_query]
    base_query_embeddings = embeddings[base_id_query]
    seen_query_embeddings = embeddings[seen_id_query]
    all_query_embeddings = torch.cat((base_query_embeddings, seen_query_embeddings, novel_query_embeddings), dim=0)

    all_query_embeddings = F.normalize(all_query_embeddings, p=2, dim=-1)
    pairwise_distance_all = F.linear(all_query_embeddings, proto_list)

    labels_all_query = (adjust_labels(labels[all_id_query], all_encounter_class, num_classes)).to(device)
    labels_all_query = labels_all_query.cpu().detach()

    all_acc_test = accuracy(pairwise_distance_all, labels_all_query)
    base_acc_test = accuracy(pairwise_distance_all[:len(base_id_query), :], labels_all_query[:len(base_id_query)])
    seen_acc_test = accuracy(pairwise_distance_all[len(base_id_query):, :], labels_all_query[len(base_id_query):])

    return all_acc_test, base_acc_test, seen_acc_test

def prototype_shift(encoder, novel_encoder, features, curr_adj, cls_prototypes_old, novel_id_support, sigma):
    encoder.eval()
    novel_encoder.eval()

    embeddings_curr, _ = novel_encoder(features, curr_adj)
    embeddings_curr = embeddings_curr[novel_id_support]
    embeddings_prior, _ = encoder(features, curr_adj)
    embeddings_prior = embeddings_prior[novel_id_support]
    delta_phi = embeddings_curr - embeddings_prior

    for i in range(cls_prototypes_old.size(0)):
        cls_feature_prototype = cls_prototypes_old[i].expand(embeddings_prior.size())
        distance = F.pairwise_distance(embeddings_prior, cls_feature_prototype, p=2)

        distance = torch.square(distance)
        divider = 2 * (sigma ** 2)
        omega = torch.exp(-(distance / divider)).view(-1, 1)
        cls_feature_drift_denominator = torch.sum(omega * delta_phi, dim=0)
        cls_feature_drift_numerator = torch.sum(omega, dim=0).clamp(1e-12)
        cls_feature_drift = cls_feature_drift_denominator / cls_feature_drift_numerator
        cls_prototypes_old[i] = cls_prototypes_old[i] + cls_feature_drift

    return cls_prototypes_old

def novel_prototypes_calibration(embeddings, curr_adj, id_support, cls_wise_prototypes,
                                 novel_id_support, novel_class_selected, novel_id_query, args):
    novel_support_embedding_list = []
    z_dim = embeddings.size(1)
    novel_support_embeddings = embeddings[id_support]

    if args.novel_calib:
        if not args.kmean_refine:
            for j, idx in enumerate(novel_id_support):
                neighs, _, _, _ = k_hop_subgraph(int(idx), 2, curr_adj)
                neighs_embeddings = embeddings[neighs].view(-1, z_dim)
                nsj = F.normalize(novel_support_embeddings[j], p=2, dim=-1)
                neighs_embeddings = F.normalize(neighs_embeddings, p=2, dim=-1)
                attn = F.linear(nsj, neighs_embeddings)
                # attn = attn / args.scale
                attn = F.softmax(attn, dim=-1)
                mix_embeddings = torch.mm(attn.view(1, -1), neighs_embeddings)
                novel_support_embedding_list.append(
                    0. * novel_support_embeddings[j].view(1, z_dim) + 1. * mix_embeddings)
            novel_support_embeddings = torch.cat(novel_support_embedding_list)

            novel_support_embeddings = novel_support_embeddings.view([len(novel_class_selected), args.shot, z_dim])
            novel_prototype_embeddings = novel_support_embeddings.mean(1)
        else:
            novel_support_embeddings = novel_support_embeddings.view([len(novel_class_selected), args.shot, z_dim])
            novel_prototype_embeddings = novel_support_embeddings.mean(1)
            novel_prototypes_t = novel_prototype_embeddings
            for k in range(args.kmean_eps):
                neighbors_list = []
                for j, idx in enumerate(novel_id_support):
                    neighs, _, _, _ = k_hop_subgraph(int(idx), args.kmean_hops, curr_adj)
                    neighbors_list.extend(neighs)
                # neighbors_list = novel_id_query
                novel_prototypes_t, _ = compute_prototypes(embeddings, [], novel_prototypes_t,
                                                           novel_id_support,
                                                           neighbors_list, novel_class_selected, args.way,
                                                            cls_wise_prototypes, gamma=args.gamma, scale=args.scale)
                novel_prototype_embeddings = novel_prototypes_t
    else:
        novel_support_embeddings = novel_support_embeddings.view([len(novel_class_selected), args.shot, z_dim])
        novel_prototype_embeddings = novel_support_embeddings.mean(1)
    return novel_prototype_embeddings

def finetune(novel_encoder, features, curr_adj, novel_id_support, novel_id_query, labels_novel_support, labels_novel_query,
                          cls_seen_prototypes, epoch, novel_class_selected, optimizer_novel_encoder, criterion, args):
    novel_encoder.train()
    optimizer_novel_encoder.zero_grad()

    cls_seen_prototypes = cls_seen_prototypes.detach()
    embeddings, _ = novel_encoder(features, curr_adj)
    novel_support_embeddings = embeddings[novel_id_support]
    z_dim = embeddings.size(1)

    if args.novel_calib:
        novel_prototype_embeddings = novel_prototypes_calibration(embeddings, curr_adj, novel_id_support,
                                                                  cls_seen_prototypes, novel_id_support,
                                                                  novel_class_selected, novel_id_query, args)
    else:
        novel_prototype_embeddings = novel_support_embeddings.view([len(novel_class_selected), args.shot, z_dim]).mean(1)

    proto_all = torch.cat((cls_seen_prototypes, novel_prototype_embeddings), dim=0)
    proto_all = F.normalize(proto_all, p=2, dim=-1)
    novel_support_embeddings = F.normalize(novel_support_embeddings, p=2, dim=-1)
    if args.loss_type == 'crossentropy':
        pairwise_distance_novel = F.linear(novel_support_embeddings, proto_all) / args.scale
    else:
        pairwise_distance_novel = F.linear(novel_support_embeddings, proto_all)

    acc_novel_support = accuracy(pairwise_distance_novel, labels_novel_support)
    angular_loss = criterion(pairwise_distance_novel, labels_novel_support)

    loss_train = angular_loss
    loss_train.backward()
    optimizer_novel_encoder.step()

    # eval
    novel_encoder.eval()
    embeddings, _ = novel_encoder(features, curr_adj)

    novel_support_embeddings = embeddings[novel_id_support]
    novel_support_embeddings = novel_support_embeddings.view([len(novel_class_selected), args.shot, embeddings.size(1)])
    novel_prototype_embeddings = novel_support_embeddings.mean(1)
    proto_all_val = torch.cat((cls_seen_prototypes, novel_prototype_embeddings), dim=0)
    proto_all_val = F.normalize(proto_all_val, p=2, dim=-1)

    novel_query_embeddings = embeddings[novel_id_query]
    pairwise_distance_novel = F.linear(novel_query_embeddings, proto_all_val)

    wf = F.log_softmax(pairwise_distance_novel, dim=-1)
    query_loss = F.nll_loss(wf, labels_novel_query)
    acc_novel_query = accuracy(pairwise_distance_novel, labels_novel_query)

    print('epoch_finetune:{}, support_loss:{:.4f}, support:{:.4f}, query_loss: {:.4f}, query_acc: {:.4f}'.format(epoch,
                                                                                                                 loss_train,
                                                                                                                 acc_novel_support,
                                                                                                                 query_loss,
                                                                                                                 acc_novel_query))
    return

def update_novel_prototypes(novel_encoder, features, curr_adj, novel_id_support, cls_wise_prototypes,
                            novel_class_selected, novel_id_query, args):
    novel_encoder.eval()
    embeddings, _ = novel_encoder(features, curr_adj)
    novel_support_embeddings = embeddings[novel_id_support]
    z_dim = embeddings.size(1)
    if args.novel_calib:
        novel_prototype_embeddings = novel_prototypes_calibration(embeddings, curr_adj, novel_id_support,
                                                                  cls_wise_prototypes, novel_id_support,
                                                                  novel_class_selected, novel_id_query, args)
    else:
        novel_support_embeddings = novel_support_embeddings.view([len(novel_class_selected), args.shot, z_dim])
        novel_prototype_embeddings = novel_support_embeddings.mean(1)
    return novel_prototype_embeddings

def update_momentum_encoder_b_to_n(base_model, novel_model, beta):
    """Momentum update of the momentum encoder"""
    for param_b, param_n in zip(base_model.parameters(), novel_model.parameters()):
        param_n.data = param_n.data * (1. - beta) + param_b.data * beta

def update_momentum_encoder_n_to_b(base_model, novel_model):
    """Momentum update of the momentum encoder"""
    for param_b, param_n in zip(base_model.parameters(), novel_model.parameters()):
        param_b.data = copy.deepcopy(param_n.data)
