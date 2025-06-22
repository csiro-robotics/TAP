import torch
import torch.nn.functional as F
import copy


def compute_prototypes(embeddings, seen_prototypes, novel_prototypes_t_1, novel_support_id, novel_query_id,
                         novel_class_id, n_way, cls_wise_prototypes, gamma=0.7, scale=0.1):
    prototypes = torch.cat((cls_wise_prototypes, novel_prototypes_t_1), dim=0)
    dist_query = F.linear(F.normalize(embeddings[novel_query_id], p=2, dim=1), F.normalize(prototypes, p=2, dim=1)) / scale
    probs = F.softmax(dist_query, dim=1)
    new_proto_list = []
    probs, pred_labels = probs.max(1)
    novel_class_id = [novel_class_id.index(j)+cls_wise_prototypes.size(0) for j in novel_class_id]
    for i in range(len(novel_class_id)):
        idx = torch.where(pred_labels==novel_class_id[i])[0]
        temp_protos = (probs[idx].view(-1, 1)*embeddings[novel_query_id][idx]).sum(0) + embeddings[novel_support_id[i*n_way:(i+1)*n_way]].sum(0)
        temp_protos = temp_protos / (probs[idx].sum()+n_way)
        new_proto_list.append(temp_protos.unsqueeze(0))

    novel_prototypes_t = gamma * torch.cat(new_proto_list, dim=0) + (1-gamma) * novel_prototypes_t_1

    return novel_prototypes_t, torch.cat(new_proto_list, dim=0)


