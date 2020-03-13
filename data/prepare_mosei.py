#!/usr/bin/env python

import os,sys
sys.path.insert(0, os.getcwd())

import pickle
import re
import numpy as np

import mmsdk
from mmsdk import mmdatasdk as md

DATA_PATH = './data/CMU_MOSEI_RAW/'

saved_dataset = './data/cmu_mosei_dataset_fill.pkl'

acoustic_field = 'CMU_MOSEI_COVAREP'
text_field = 'CMU_MOSEI_TimestampedWords'
label_field_emo = 'CMU_MOSEI_LabelsEmotions'
label_field_sen = 'CMU_MOSEI_LabelsSentiment'

# Maximum frame size of acoustic features
max_feat_len = 200


# create folders for storing the data
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# download CMU-MOSEI
# DATASET = md.cmu_mosei
dataset_cmu_mosei = {}
dataset_cmu_mosei["raw"] = {
"words": 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedWords.csd'
}

dataset_cmu_mosei["highlevel"] = {
"COVAREP": 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/CMU_MOSEI_COVAREP.csd'
}

dataset_cmu_mosei["labels"] = {
"Sentiment Labels": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_LabelsSentiment.csd",
"Emotion Labels": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_LabelsEmotions.csd"
}

try:
    md.mmdataset(dataset_cmu_mosei["highlevel"], DATA_PATH)
except RuntimeError:
    print("High-level features have been downloaded previously.")

try:
    md.mmdataset(dataset_cmu_mosei["raw"], DATA_PATH)
except RuntimeError:
    print("Raw data have been downloaded previously.")
    
try:
    md.mmdataset(dataset_cmu_mosei["labels"], DATA_PATH)
except RuntimeError:
    print("Labels have been downloaded previously.")


features = [
    text_field, 
    acoustic_field
]

recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
dataset = md.mmdataset(recipe)


# we define a simple averaging function that does not depend on intervals
def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features

def fill(intervals: np.array, features: np.array) -> np.array:
    try:
        if features.shape[1] != 74 :
            return avg(intervals, features)

        filled = np.zeros((max_feat_len, features.shape[1]), dtype=features.dtype)
        end = min(features.shape[0], max_feat_len)
        filled[:end, :] = features[:end,:]
        filled = filled.reshape(-1)
        return filled 
    except:
        return features

# first we align to words with averaging, collapse_function receives a list of functions
dataset.align(text_field, collapse_functions=[fill])

label_recipe_emo = {label_field_emo: os.path.join(DATA_PATH, label_field_emo + '.csd')}
dataset.add_computational_sequences(label_recipe_emo, destination=None)
dataset.align(label_field_emo)

# label_recipe_sen = {label_field_sen: os.path.join(DATA_PATH, label_field_sen + '.csd')}
# dataset.add_computational_sequences(label_recipe_sen, destination=None)
# dataset.align(label_field_sen)

pickle.dump(dataset, open(saved_dataset,"wb"))

