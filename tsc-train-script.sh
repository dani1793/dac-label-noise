#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:48:36 2020

@author: daniyalusmani1
"""


python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-resnet --rand_labels 0.3 --epochs 100 --loss_fn dac_loss --batch_size 128 --test_batch_size 128  --save_best_model --seed 0