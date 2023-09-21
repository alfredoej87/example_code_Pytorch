import torch
import copy
import time
import torch.utils.data
from torch import nn, Tensor, autograd
import torch.nn.functional as F
import codecs
from torch.nn.utils.rnn import pack_padded_sequence
from math import log2
import logging
import matplotlib
import io
import torchaudio
import numpy
import torchaudio.transforms as T
import random

import soundfile as sf
from sklearn.datasets import make_moons

matplotlib.use('Agg')
from collections import OrderedDict
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
import datasets as ds
from datasets import load_dataset, Audio
import matplotlib.pyplot as plt
from common.utilities import load_wav_to_torch, load_wav_to_torch3
from transformers import HubertModel, HubertConfig, AutoProcessor, Wav2Vec2Processor

from typing import Optional, Any, Union, Callable
import sys
import gc
from torchsummary import summary
from scipy.special import rel_entr
# from hparams import create_hparams
from typing import Optional, Any, Union, Callable
import sys
from scipy.special import rel_entr
import numpy as np
import pandas as pd
import seaborn as sns
import math
import argparse
import re
import os
import json
import sys
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn import ModuleList, Module
from huggingface_hub import hf_hub_url, cached_download

_SAMPLE_DIR = "LJSpeech-1.1/wavs"
_SAMPLE_DIR2 = "LJSpeech-1.1/mels/mels"
_HUBERT_DIR = "LJSpeech-1.1/hubertgrams"
configuration = HubertConfig()
model = HubertModel(configuration)
configuration = model.config

lj_dataset = ds.load_dataset('lj_speech', split="train")

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")  
resampler = torchaudio.transforms.Resample(22_050, 16_000)
torch.cuda.empty_cache()

def map_to_array(batch):
     speech_array, _ = sf.read(batch["file"]) ####+".wav")
     batch["speech"] = speech_array
     return batch

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["file"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

file_to_query2 = open('file_repository_frames_allInfo22050Hz.json')  
trainingFilesRepository = json.loads(file_to_query2.read())

lj_dataset = lj_dataset.map(speech_file_to_array_fn)   
megaHubertFeatures = np.ndarray(shape = (0,128))
pca = PCA(n_components=128)

meanAllLJ = numpy.zeros((1024,))
accumTotalFeatures = 0

###1st retrieve mean
for numFile in range(13100):   ##Compute only from 10K elements (77% of the data)
    speechSeeked = lj_dataset[numFile]["speech"]
    speechAudioID = lj_dataset[numFile]["id"]
    print("Processing file number ", numFile, "with AudioID ", speechAudioID)
    inputValues = processor(speechSeeked, sampling_rate = 16_000, return_tensors = "pt").input_values
    hiddenStates = model(inputValues).last_hidden_state   ###these are produced at 16 kHz at a frame rate of 20 ms
    hubertFeatures = hiddenStates.squeeze(0).t()  ###E.g., shape produced is (445, 1024)
    hubertFeatures2 = hiddenStates.squeeze(0)
    savedHubertName = speechAudioID+".pt"
    ###Transpose to get e.g., (1024, 445) based features --> so have analogous shape as for mels (80,xx)
    ###Determine the required number of features in 22 kHz and with frame rate 11.61 ms (for this read the info of the filesRepository)
    requiredNumFeatures = trainingFilesRepository[speechAudioID]['numOfmels']
    hubertFeaturesExpanded = torch.zeros(1024, requiredNumFeatures)
    requiredRatio = hubertFeatures.shape[1]/requiredNumFeatures
    for numFeature in range(requiredNumFeatures):
        idxFromPicking = round((numFeature+1)*requiredRatio)-1
        hubertFeaturesExpanded[:,numFeature] = hubertFeatures[:,idxFromPicking]
    hubertFeatsArry = hubertFeaturesExpanded.t().detach().numpy()
    hubertForCovariance = hubertFeaturesExpanded.detach().numpy()
    ###Need to obtain a mean of 1024 elements (so that each feature is transformed based on that)
    currentMean = numpy.mean(hubertForCovariance, axis = 1)
    accumTotalFeatures +=requiredNumFeatures
    meanAllLJ += requiredNumFeatures*currentMean

meanAllLJ /=(accumTotalFeatures)
print("mean of all Features ", meanAllLJ)
print("Accum total features ", accumTotalFeatures)   ###we will obtain a very big number
print("mean of all features", meanAllLJ.shape)
print("Now will obtain covariance")
covarianceAllLJ = numpy.zeros((1024, 1024))
###Having obtained the mean, it is time to update the covariance matrix
accumTotalFeatures = 0  ###Careful to re-initialize
for numFile in range(13100):   
    speechSeeked = lj_dataset[numFile]["speech"]
    speechAudioID = lj_dataset[numFile]["id"]   
    print("Processing file number ", numFile, "with AudioID ", speechAudioID)
    inputValues = processor(speechSeeked, sampling_rate = 16_000, return_tensors = "pt").input_values    
    hiddenStates = model(inputValues).last_hidden_state   ###these are produced at 16 kHz at a frame rate of 20 ms
    hubertFeatures = hiddenStates.squeeze(0).t()  ###E.g., shape produced is (445, 1024)
    hubertFeatures2 = hiddenStates.squeeze(0)
    savedHubertName = speechAudioID+".pt"
    ###Transpose to get e.g., (1024, 445) based features --> so have analogous shape as for mels (80,xx)  
    ###Determine the required number of features in 22 kHz and with frame rate 11.61 ms (for this read the info of the filesRepository)
    requiredNumFeatures = trainingFilesRepository[speechAudioID]['numOfmels']    
    hubertFeaturesExpanded = torch.zeros(1024, requiredNumFeatures)
    requiredRatio = hubertFeatures.shape[1]/requiredNumFeatures
    for numFeature in range(requiredNumFeatures):
        idxFromPicking = round((numFeature+1)*requiredRatio)-1
        hubertFeaturesExpanded[:,numFeature] = hubertFeatures[:,idxFromPicking]
    hubertFeatsArry = hubertFeaturesExpanded.t().detach().numpy()
    hubertForCovariance = hubertFeaturesExpanded.detach().numpy()
    hubertCentered = hubertForCovariance-meanAllLJ[:,np.newaxis]
    currentCovariance = hubertCentered.dot(np.transpose(hubertCentered))
    ###Need to obtain a mean of 1024 elements (so that each feature is transformed based on that)
    covarianceAllLJ += currentCovariance
    accumTotalFeatures +=requiredNumFeatures
    
covarianceAllLJ /= (accumTotalFeatures-1)   ###so that it is an unbiased estimator by (N-1) instead of (N)
print("Shape of covariance is ", covarianceAllLJ.shape)

eigVals, eigVecs = numpy.linalg.eig(covarianceAllLJ)
#print(eigVals)
sortedIdx = numpy.argsort(eigVals)[::-1]
#print(sortedIdx) 
sorted_eigVals = eigVals[sortedIdx]
#print("sorted eigenvalues ", sorted_eigVals[0:11])
sorted_eigVecs = eigVecs[sortedIdx]    
explainedVarRatio = sorted_eigVals/numpy.sum(sorted_eigVals)
#print("explained ratio ", explainedVarRatio[0:45])
#print("total expalined by 1st 46 ", numpy.sum(explainedVarRatio[0:45]))
#print("shape of eigenvectors ", sorted_eigVecs.shape)
selectedTransform = sorted_eigVecs[:,:128]
#print("shape of transform ", selectedTransform.shape)

print("NOW REDUCING DIMENSIONALITY")
##After obtaining the eigenvectors, transform the data in an individual way
for numFile in range(13100):
    speechSeeked = lj_dataset[numFile]["speech"]
    speechAudioID = lj_dataset[numFile]["id"]
 #   print("********NOW PROJECTING OUR DATA USING PCA AND LESS DIMENSIONS *****************!!!")
    print("Processing file number ", numFile, "with AudioID ", speechAudioID)
    inputValues = processor(speechSeeked, sampling_rate = 16_000, return_tensors = "pt").input_values
    hiddenStates = model(inputValues).last_hidden_state   ###these are produced at 16 kHz at a frame rate of 20 ms
    hubertFeatures = hiddenStates.squeeze(0).t()  ###E.g., shape produced is (445, 1024)
    hubertFeatures2 = hiddenStates.squeeze(0)
    savedHubertName = speechAudioID+".pt"
    requiredNumFeatures = trainingFilesRepository[speechAudioID]['numOfmels']
    hubertFeaturesExpanded = torch.zeros(1024, requiredNumFeatures)
    requiredRatio = hubertFeatures.shape[1]/requiredNumFeatures
    for numFeature in range(requiredNumFeatures):
        idxFromPicking = round((numFeature+1)*requiredRatio)-1
        hubertFeaturesExpanded[:,numFeature] = hubertFeatures[:,idxFromPicking]
    hubertCentered = hubertFeaturesExpanded.t().detach().numpy()-meanAllLJ[numpy.newaxis,:]   ###Before doing PCA, the data needs to be centered
    #hubertFeatsArry = hubertCentered.detach().numpy()  ##(832, 1024)
    transformedPCAfeatures = hubertCentered.dot(selectedTransform)
    hubertFeatures128Dim = torch.from_numpy(transformedPCAfeatures).t()  ###We need to transpose to match our implementation in terms of mel-spectrograms
    torch.save(hubertFeatures128Dim, os.path.join(_HUBERT_DIR, savedHubertName))   
 
