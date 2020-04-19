#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:26:04 2020

@author: daniyalusmani1
"""

import pandas as pd
import argparse

import numpy as np
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser(description='Plot creation for TSC DAC architecture',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--orig_dataset', default=None, type=str, help='original dataset which in which noise was introduced')
parser.add_argument('--noisy_dataset', default=None, type=str, help='noisy dataset which in which noise was introduced')
parser.add_argument('--noisy_percentage', default=0, type=float, help='noisy percentage')
parser.add_argument('--dataset', default=None, type=str, help='Name of dataset')


args = parser.parse_args()
def generateOriginalDatasetClassDist():
    
    orig = pd.read_csv(args.orig_dataset, sep='\t', header=None)
    origLabels = orig[0] - 1
    classes = len(np.unique(origLabels)) 
    fig, ax = plt.subplots()
    
    n, bins, patches = ax.hist(origLabels, density=1, edgecolor='black', bins=classes, linewidth=1.2)
    plt.xlabel('Classes')
    plt.ylabel('count')
    plt.title('Class distribution in original %s set'%(args.dataset))
    
    plt.savefig('results/plots/%s-original-class-dist.png'%(args.dataset))
    plt.close()

def generateNoisyDatasetClassDist():
    rootPath = args.noisy_dataset + '_' + str(args.noisy_percentage)
    noisyTrain = pd.read_csv(rootPath +  '_TRAIN.tsv', sep='\t', header=None)
    noisyVal = pd.read_csv(rootPath + '_VAL.tsv', sep='\t', header=None)
    noisyLabels = np.append(noisyTrain[0], noisyVal[0])
    classes = len(np.unique(noisyLabels)) 
    fig, ax = plt.subplots()
    
    n, bins, patches = ax.hist(noisyLabels, density=1, edgecolor='black', bins=classes, linewidth=1.2)
    plt.xlabel('Classes')
    plt.ylabel('count')
    plt.title('Class distribution in noisy %s set'%(args.dataset))
    
    plt.savefig('results/plots/%s-noisy-%s-class-dist.png'%(args.dataset, str(args.noisy_percentage)))
    plt.close()
    
def _extractAccumuatedActivations(path, files):
    activations = []
    for file in files:
        epoch = np.load(path + '/' + file,  allow_pickle=True);
        if len(activations) == 0:
            activations = epoch.mean(axis=0);
            activations = np.expand_dims(activations, axis = 1)
        else:
            activations = np.append(activations, np.expand_dims(epoch.mean(axis=0), axis = 1), axis = 1)
        # print(activations.shape)
    return activations

def _plotAccumulatedActivations(activations, activationType):
    for c in range(activations.shape[0]):
        if c == activations.shape[0] - 1:
            plt.plot(activations[c], label = '%s'%('Abstention'))
        else:
            plt.plot(activations[c], label = 'class %s'%(str(c)))
    plt.title('%s Activation energies for %s with noise %s'%(activationType, args.dataset, str(args.noisy_percentage)))
    plt.xlabel('Epochs')
    plt.ylabel('mean activation energy / epoch')
    plt.legend(loc='upper right', fontsize='xx-small')
    # plt.savefig('results/plots/%s-%s-noisy-%s-activations.png'%(args.dataset, activationType, str(args.noisy_percentage)), dpi = 1080)
    plt.savefig('results/plots/%s-%s-noisy-%s-activations.png'%(args.dataset, activationType, str(args.noisy_percentage)), dpi = 1080, format='jpeg')
    plt.close()

def generateActivationEnergyPlotsPerEpoch():
    rootPath = 'results/' + args.dataset.lower() + '/' + str(args.noisy_percentage)
    script_dir = os.path.dirname(os.path.realpath('__file__'))
   # script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rootPath)
    print(abs_file_path)
    entries = os.listdir(abs_file_path)
    entries = np.array(entries)
    trainIndices = ['train' in entry for entry in entries]
    valIndices = [not 'train' in entry for entry in entries]
    valActivations = entries[valIndices]
    trainActivations = entries[trainIndices]
    
    accumlatedTrainActivations = _extractAccumuatedActivations(abs_file_path, trainActivations)
    accumlatedValActivations = _extractAccumuatedActivations(abs_file_path, valActivations)
    
    _plotAccumulatedActivations(accumlatedTrainActivations, 'train')
    _plotAccumulatedActivations(accumlatedValActivations, 'val')

    
    
    

#generateOriginalDatasetClassDist()
#generateNoisyDatasetClassDist()

generateActivationEnergyPlotsPerEpoch()
