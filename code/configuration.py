import argparse

def update_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:/Amazon_clothing/dblp/cora_full')
    parser.add_argument('--use_cuda', default=True, help='Enable CUDA training.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train.')
    parser.add_argument('--episodes_ft', type=int, default=5, help='Number of episodes to finetune.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--way', type=int, default=5, help='way.')
    parser.add_argument('--shot', type=int, default=5, help='shot.')
    parser.add_argument('--loss_type', type=str, default='cosface',
                        choices=['arcface', 'sphereface', 'cosface', 'crossentropy'])
    parser.add_argument('--session', type=int, default=9, help='sessions.')

    parser.add_argument('--eye_pertb', type=int, default=1, help='if use the TFA aug')
    parser.add_argument('--pertb', type=int, default=1, help='if use the TVA augmentation.')
    parser.add_argument('--finetune', type=int, default=1, help='if do model finetune on novel classes')
    parser.add_argument('--novel_calib', type=int, default=1, help='if update novel prototypes')
    parser.add_argument('--kmean_refine', type=int, default=1,
                        help='if use kmeans or attention to update novel prototypes')
    parser.add_argument('--kmean_hops', type=int, default=2, help='hops for novel prototype update')
    parser.add_argument('--kmean_eps', type=int, default=2, help='2, steps for novel prototype update')

    parser.add_argument('--gat_heads', type=int, default=12, help='number of gat heads')
    parser.add_argument('--alpha', type=float, default=0.7, help='loss weight for augmentation data')
    parser.add_argument('--beta', type=float, default=0.95, help='moving average rate for EMA')
    parser.add_argument('--tau', type=float, default=15.0)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=1., help='coefficient of PSO')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='moving average rate for kmeans prototype calibration in IPCN')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for novel support mix neighbors')

    args = parser.parse_args()

    return args