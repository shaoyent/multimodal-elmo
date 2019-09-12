import pickle
import re

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class DNN(nn.Module):
    def __init__(self, options):
        super(DNN, self).__init__()

        mixture_size = 2
        trainable = True
        initial_scalar_parameters = [0, 1]
        dropout = 0.2
        # activation = nn.ReLU
        activation = nn.Tanh

        self.input_size = options['input_size']
        self.num_classes = options['num_classes']

        self.make_layers(options)

        self.fc = nn.Sequential( *self.layers)

        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)
        self.scalar_parameters = nn.ParameterList(
                        [nn.Parameter(torch.FloatTensor([initial_scalar_parameters[i]]),
                                           requires_grad=trainable) for i in range(mixture_size)])

    def make_layers(self, options):
        # self.layers = nn.ModuleList()
        self.layers = []
        if options['activation'] == 'relu' :
            activation = nn.ReLU
        elif options['activation'] == 'tanh' :
            activation = nn.Tanh

        in_dim = self.input_size
        for dim in options['layers'] :

            self.layers.append(nn.Dropout(p=options['dropout']))
            self.layers.append(nn.Linear(in_dim, dim))
            self.layers.append(activation())

            in_dim = dim

        self.layers.append(nn.Dropout(p=options['dropout']))
        self.layers.append(nn.Linear(in_dim, self.num_classes))


    def forward(self, x):

        normed_weights = F.softmax(torch.cat([parameter for parameter
                                    in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        tensors = torch.split(x, split_size_or_sections=1, dim=1)

        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)

        x = self.gamma * sum(pieces)
        x = x.squeeze(dim=1)
        x = self.fc(x)

        # x = torch.sigmoid(x)
        # x = F.relu(self.final_layer(x))

        # x = self.final_layer(x)

        return x


class mosei_dataset(Dataset):
    def __init__(self, dataset, embedding_key='embeddings', splits=None, oversample=False, min_ir=1, fn_labels=None):
        pattern = re.compile('(.*)\[.*\]')

        self.min_ir = min_ir

        self.dataset = {
            'embeddings': {},
            'labels': {} 
        }

        if fn_labels is not None:
            labels = pickle.load(open(fn_labels, "rb"))

        self.keys = []

        if splits is not None :
            for segment in dataset["labels"].keys(): 
                vid_id = re.search(pattern, segment).group(1)
                if vid_id in splits:
                    if isinstance(dataset['labels'][segment], dict) :
                        self.dataset['labels'][segment] = dataset['labels'][segment]['features']
                    else :
                        self.dataset['labels'][segment] = dataset['labels'][segment]
                    self.dataset['embeddings'][segment] = dataset[embedding_key][segment]
                    self.keys.append(segment)
                    # self.dataset['labels'][segment] = labels["CMU_MOSEI_LabelsSentiment"][segment]

        if oversample :
            self.balance_classes()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        embedding_0 = torch.from_numpy(self.dataset['embeddings'][key][0]).unsqueeze(dim=0)
        embedding_1 = torch.from_numpy(self.dataset['embeddings'][key][1]).unsqueeze(dim=0)

        embedding = torch.cat((embedding_0, embedding_1), dim=0)
        # label = torch.from_numpy(self.dataset['labels'][key]) + 3
        label = np.greater(self.dataset['labels'][key], 0).astype(np.float32).reshape(-1,)

        no_emotion = (label.sum(axis=-1) == 0).astype(np.float32).reshape(-1,)
        # label = np.concatenate((label, no_emotion), axis=-1)

        label = torch.from_numpy(label)

        return embedding, label

    def balance_classes(self):

        oversample = []
        
        def imbalance_ratio():
            labels = np.concatenate(list(self.dataset['labels'].values()), axis=0)
            labels = labels > 0

            class_sum = np.sum(labels, axis=0)

            IRpl = max(class_sum) / class_sum
            meanIR = IRpl.mean()

            return IRpl, meanIR

        IRpl, meanIR = imbalance_ratio()
        aug = 1 
        while meanIR > self.min_ir :
            y = np.argmax(IRpl)
            ym = np.argmin(IRpl)
            random.shuffle(self.keys)
            found = 0 
            new_keys = []
            for key in self.keys :
                if "_aug" in key : 
                    continue
                if  self.dataset['labels'][key][0,y] > 0 :
                    if  found < 5 and self.dataset['labels'][key][0,ym] > 0 :
                    # if  self.dataset['labels'][key][0,ym] > 0 :
                        continue
                    new_key = f'{key}_aug{y}_{aug}'
                    self.dataset['labels'][new_key] = self.dataset['labels'][key]
                    self.dataset['embeddings'][new_key] = self.dataset['embeddings'][key]
                    new_keys.append(new_key)

                    aug += 1
                    found += 1

                if found > 10 :
                    self.keys.extend(new_keys)
                    break
            IRpl, meanIR = imbalance_ratio()
            # print(IRpl, meanIR, y, len(self.keys))
         

