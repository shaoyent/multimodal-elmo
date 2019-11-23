from __future__ import print_function

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import json

import pickle

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import re
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSEI import cmu_mosei_std_folds

from utils import *

train_split = cmu_mosei_std_folds.standard_train_fold
dev_split = cmu_mosei_std_folds.standard_valid_fold
test_split = cmu_mosei_std_folds.standard_test_fold

options = {
    'input_size': 1024,
    'optimizer': 'Adam',
    'num_classes': 6,
    'batch_size': 20,
    'class_weights': [1, 1, 1, 1, 1, 1],
    'layers': [1000,500], 
    'activation': 'relu',
    'dropout': 0.4,
    'num_epochs': 20,
    'gamma': 1.6, 
    'min_ir': 2.8, 
}

def main():

    parser = argparse.ArgumentParser(description='Emotion recognition on CMU-MOSEI')
    parser.add_argument('--seed', default=5566, type=int, help="Random seed")
    parser.add_argument('--save_dir', default='./results', type=str, help="")
    parser.add_argument('--num_epochs', default=20, type=int, help="Number of training epochs")
    parser.add_argument('--dropout', default=0.5, type=float, help="")
    parser.add_argument('--min_ir', default=2, type=float, help="Minimum imbalance ratio")
    parser.add_argument('--lr', default=0.5, type=float, help="")
    parser.add_argument('--activation', default='relu', type=str, help="")
    parser.add_argument('--batch_size', default=32, type=int, help="")
    parser.add_argument('--layers', default='512.512.256.256.128.128', type=str, help="Comma-separted list of hidden dimensions")
    parser.add_argument('--gamma', default=1, type=float, help="Weight for negative class")
    parser.add_argument('--dataset', default=None, type=str, help="Dataset")
    parser.add_argument('--verbose', default=False, action='store_true', help="Verbose")

    args = parser.parse_args()

    options['num_epochs'] = args.num_epochs
    options['dropout'] = args.dropout
    options['activation'] = args.activation
    options['batch_size'] = args.batch_size
    options['layers'] = [ int(x) for x in args.layers.split('.')]
    options['gamma'] = args.gamma
    options['lr'] = args.lr
    options['min_ir'] = args.min_ir

    grad_clip_value = 10.0

    CUDA = True
    import torch
    torch.manual_seed(args.seed)
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, Adagrad
    
    from torch.utils.data import Dataset, DataLoader
    
    from losses import WeightedBCELoss
    from models import DNN, mosei_dataset


    n_epochs = options['num_epochs']
    batch_size = options['batch_size']
    verbose = args.verbose

    save_dir = args.save_dir
    ckpt_path = os.path.join(save_dir, "checkpoint.pt")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir, 'options.json'), 'w') as fout:
        fout.write(json.dumps(options))
    
    class_weight = torch.Tensor(options['class_weights'])
    class_weight = class_weight / torch.sum(class_weight)
    gamma = torch.Tensor([options['gamma']])

    if CUDA :
        class_weight = class_weight.cuda()
        gamma = gamma.cuda()

    
    model = DNN(options)
    if CUDA:
        model.cuda()
    if verbose: print(model)

    model.train()

    dataset = pickle.load(open(args.dataset, "rb"))

    train_dataset = mosei_dataset(dataset, splits=train_split, oversample=True, min_ir=options['min_ir'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    if verbose: print(f"Train set: {len(train_dataset)} samples")

    val_dataset = mosei_dataset(dataset, splits=dev_split)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    if verbose: print(f"Val set: {len(val_dataset)} samples")

    test_dataset = mosei_dataset(dataset, splits=test_split)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    if verbose: print(f"Test set: {len(test_dataset)} samples")

    optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=options['lr'])

    criterion = WeightedBCELoss(class_weight=class_weight, PosWeightIsDynamic=True, gamma=gamma)

    best_val = np.Inf 
    best_metric = 0 

    train_labels = []

        
    for epoch_no in range(n_epochs):
        total_pos = 0 
        model.train()
        for batch_no, batch in enumerate(train_loader, start=1):
            embeddings, labels = batch
            if CUDA:
                embeddings, labels = embeddings.cuda(), labels.cuda()

            y_hat = model(embeddings)
            loss = criterion(y_hat, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
            optimizer.step()
            optimizer.zero_grad()

            if batch_no % 200 == 0 : 
                if verbose: print(f"Training loss: {loss.item():.5f}")

                y_true, y_pred, val_loss = [], [], []
                model.eval()
                for batch in val_loader: 
                    embeddings, labels = batch
                    if CUDA:
                        embeddings, labels = embeddings.cuda(), labels.cuda()
                    y_hat = model(embeddings)
                    loss = criterion(y_hat, labels)
                    val_loss.append(loss.item())

                    y_true.append(labels.detach().cpu().numpy())
                    y_pred.append(y_hat.detach().cpu().numpy())

                    assert not np.any(np.isnan(val_loss))

                y_true = np.concatenate(y_true, axis=0).squeeze()
                y_pred = np.concatenate(y_pred, axis=0).squeeze()
                y_true_bin = y_true > 0
                y_pred_bin = y_pred > 0

                val_loss =  np.average(val_loss)
                f1score = [f1_score(t, p, average="weighted") for t,p in zip(y_true_bin.T, y_pred_bin.T)]
                wa = np.average(weighted_accuracy(y_true_bin, y_pred_bin))

                val_metric = np.average(f1score) + np.average(wa)
                f1score = [f'{x*100:.2f}' for x in f1score]
                if verbose: print(f"Validation loss: {val_loss:.3f}")

                if best_metric < val_metric :
                    if verbose: print("Validation metric improved")
                    best_metric = val_metric
                    checkpoint = {
                        'options': options,
                        'model': model, 
                        'epoch': epoch_no
                    }
                    torch.save(checkpoint, ckpt_path)

                model.train()

    # ====================================================================================================
    # Final Validation 
    # ====================================================================================================

    checkpoint = torch.load(ckpt_path)
    model = checkpoint['model']
    if verbose: print("Loaded best model from epoch {}".format(checkpoint['epoch']))
    model.eval()

    val_true = []
    val_pred = []

    for batch in val_loader: 
        embeddings, labels = batch
        if CUDA:
            embeddings, labels = embeddings.cuda(), labels.cuda()
        val_hat = model(embeddings)

        val_true.append(labels.detach().cpu().numpy())
        val_pred.append(val_hat.detach().cpu().numpy())

    val_true = np.concatenate(val_true, axis=0).squeeze()
    val_pred = np.concatenate(val_pred, axis=0).squeeze()

    val_true_bin = val_true > 0
    val_pred_bin = val_pred > 0

    wa = [weighted_accuracy(t, p)*100 for t,p in zip(val_true_bin.T, val_pred_bin.T)]
    f1score = [f1_score(t, p, average="weighted")*100 for t,p in zip(val_true_bin.T, val_pred_bin.T)]
    
    if verbose: print(f"Val WA, {reformat_array(wa)} Avg: {np.average(wa):.2f}")
    if verbose: print(f"Val F1, {reformat_array(f1score)} Avg: {np.average(f1score):.2f} ")
    
    # ====================================================================================================
    # Final Test 
    # ====================================================================================================
    test_true = []
    test_pred = []

    for batch in test_loader:
        embeddings, labels = batch
        if CUDA:
            embeddings, labels = embeddings.cuda(), labels.cuda()

        pred = model(embeddings)

        test_true.append(labels.detach().cpu().numpy())
        test_pred.append(pred.detach().cpu().numpy())
        

    test_true = np.concatenate(test_true, axis=0).squeeze()
    test_pred = np.concatenate(test_pred, axis=0).squeeze()

    test_true_bin = test_true > 0 # Binarized
    test_pred_bin = test_pred > 0 # Logit outputs

    test_wa = [weighted_accuracy(t, p) for t,p in zip(test_true_bin.T, test_pred_bin.T)]
    test_wa = reorder_labels(test_wa)

    test_f1score = [f1_score(t, p, average="weighted") for t,p in zip(test_true_bin.T, test_pred_bin.T)]
    test_f1score = reorder_labels(test_f1score)

    test_wa_str = [f'{x*100:.2f}' for x in test_wa]
    test_f1score_str = [f'{x*100:.2f}' for x in test_f1score]

    print(f"Test WA: {test_wa_str} Avg: {np.average(test_wa)*100:2.1f}")
    print(f"Test F1: {test_f1score_str} Avg: {np.average(test_f1score)*100:2.1f}")

    combined = [f" {x} & {y} " for x,y in zip(test_wa_str, test_f1score_str)]
    combined.append(f" {np.average(test_wa)*100:2.1f} & {np.average(test_f1score)*100:2.1f}")
    if verbose: print("&".join(combined))
   
if __name__ == "__main__":
    main()
