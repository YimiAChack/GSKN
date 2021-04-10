# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cyanure import LinearSVC
import datetime

from gckn.models import GCKNetFeature
from gckn.data import load_data, PathLoader

def eval(args, features, labels):
    features, labels = features.numpy(), labels.numpy()

    if not os.path.exists("./result/"):
        try:
            os.makedirs("./result/")
        except:
            pass

    np.savetxt("./result/gskn_" + args.dataset + ".txt", features)
    np.savetxt("./result/labels.txt", labels)

    print('Cross validation')
    train_fold_idx = [np.loadtxt('./dataset/{}/10fold_idx/train_idx-{}.txt'.format(
        args.dataset, i)).astype(int) for i in range(1, 11)]
    test_fold_idx = [np.loadtxt('./dataset/{}/10fold_idx/test_idx-{}.txt'.format(
        args.dataset, i)).astype(int) for i in range(1, 11)]
    cv_idx = zip(train_fold_idx, test_fold_idx)

    C_list = np.logspace(-4, 4, 60)
    svc = LinearSVC(C=1.0)
    clf = GridSearchCV(make_pipeline(StandardScaler(), svc),
                       {'linearsvc__C' : C_list},
                       cv=cv_idx,
                       n_jobs=4, verbose=0, return_train_score=True)
    
    clf.fit(features, labels)
    df = pd.DataFrame({'C': C_list, 
                       'train': clf.cv_results_['mean_train_score'], 
                       'test': clf.cv_results_['mean_test_score'],
                       'test_std': clf.cv_results_['std_test_score']}, 
                        columns=['C', 'train', 'test', 'test_std'])
    print(df)

    print("best test acc: %f" % (df['test'].max()))


def load_args():
    parser = argparse.ArgumentParser(
        description='Unsupervised GCKN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--path-size', type=int, nargs='+', default=[2],
                        help='path sizes for layers')
    parser.add_argument('--hidden_size', type=int, nargs='+', default=[32],
                        help='number of filters for layers')

    parser.add_argument('--anonymous_walk_length', type=int, default=10, help=' ') 
    parser.add_argument('--anonymous_walks_per_node', type=int, default=30, help=' ') 

    parser.add_argument('--pooling', type=str, default='sum',
                        help='local path pooling for each node')
    parser.add_argument('--global-pooling', type=str, default='sum',
                        help='global node pooling for each graph')
    parser.add_argument('--aggregation', action='store_true',
                        help='aggregate all path features until path size')
    parser.add_argument('--sigma', type=float, nargs='+', default=[0.5],
                        help='sigma of exponential (Gaussian) kernels for layers')
    parser.add_argument('--sampling-paths', type=int, default=300000,
                        help='number of paths to sample for unsupervised training')
    parser.add_argument('--walk', action='store_true',
                        help='use walk instead of path')
    parser.add_argument('--outdir', type=str, default='/nas/user/qk/GCKN/res',
                        help='output path')
    args = parser.parse_args()
    args.continuous = False
    if args.dataset in ['IMDBBINARY', 'IMDBMULTI', 'COLLAB']:
        # social network
        degree_as_tag = True
    elif args.dataset in ['MUTAG', 'PROTEINS', 'PTC', 'NCI1', 'sys']:
        # bioinformatics
        degree_as_tag = False
    elif args.dataset in ['BZR', 'COX2', 'ENZYMES', 'PROTEINS_full']:
        degree_as_tag = False
        args.continuous = True
    else:
        raise ValueError("Unrecognized dataset!")
    args.degree_as_tag = degree_as_tag

    args.save_logs = False
    return args


def main():
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    start_time = datetime.datetime.now()
    graphs, n_class = load_data(args.dataset, './dataset', degree_as_tag=args.degree_as_tag)
    if args.continuous:
        print("Dataset with continuous node attributes")
        node_features = np.concatenate([g.node_features for g in graphs], axis=0)
        sc = StandardScaler()
        sc.fit(node_features)
        for g in graphs:
            node_features = sc.transform(g.node_features)
            g.node_features = node_features / np.linalg.norm(node_features, axis=-1, keepdims=True).clip(min=1e-06)

    data_loader = PathLoader(graphs, max(args.path_size), args.batch_size, args.anonymous_walk_length,
                             args.anonymous_walks_per_node,
                             True, dataset=args.dataset, walk=args.walk)
    print('Computing paths...')
    if args.dataset != 'COLLAB' or max(args.path_size) <= 2: 
        data_loader.get_all_paths()
  
        
    input_size = data_loader.input_size

    print('Unsupervised training...')
    model = GCKNetFeature(input_size, args.hidden_size, args.path_size, args.anonymous_walk_length,
                          kernel_args_list=args.sigma, pooling=args.pooling,
                          global_pooling=args.global_pooling,
                          aggregation=args.aggregation)
    
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    model.module.unsup_train(data_loader, n_sampling_paths=args.sampling_paths)

    print('Encoding...')
    features, labels = model.module.predict(data_loader)

    print("Evaluating...")
    eval(args, features, labels)
    
if __name__ == "__main__":
    main()
