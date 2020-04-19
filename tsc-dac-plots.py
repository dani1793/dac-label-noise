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
    

generateOriginalDatasetClassDist()
generateNoisyDatasetClassDist()