import os
from tqdm import *
import torch
import numpy as np
import torch.utils.data as data
import mrcfile as mf
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import json
from tqdm import *
import math
import glob
import copy

def get_rawdata_list(path):
    file_list = os.listdir(path)
    mrc_list = []
    for file in file_list:
        if file.endswith('.mrc'):
            mrc_list.append(file)
    mrc_list.sort()
    
    return mrc_list


def crop_data(mrc_data, shape, padding, cval):
    len_y = math.ceil(mrc_data.shape[0]/shape[0])
    len_x = math.ceil(mrc_data.shape[1]/shape[1])
    subsets =  np.zeros((len_x*len_y, shape[0]+2*padding, shape[1]+2*padding))
    subsets.fill(cval)
    matchs = []
    subsets_index = 0
    # for j in tqdm(range(len_y),file=log_file):
    for j in range(len_y):
        for k in range(len_x):
            subset = np.zeros((shape[0]+2*padding, shape[1]+2*padding))
            subset.fill(cval)
            sj = max(0, j*shape[0]-padding)
            ej = min(mrc_data.shape[0], (j+1)*shape[0]+padding)
            sk = max(0, k*shape[1]-padding)
            ek = min(mrc_data.shape[1], (k+1)*shape[1]+padding)


            sjc = padding - j*shape[0] + sj
            ejc = sjc + (ej - sj)
            skc = padding - k*shape[1] + sk
            ekc = skc + (ek - sk)

            subset[sjc:ejc,skc:ekc] = mrc_data[sj:ej,sk:ek]
       
            subsets[subsets_index] = subset
            subsets_index += 1
            matchs.append([sj,ej,sk,ek,sjc,ejc,skc,ekc])
    return subsets, (len_y,len_x), matchs 


def concat_data(subsets, len_yx, match, padding):
    shape_y = subsets[0].shape[0]
    shape_x = subsets[0].shape[1]
    data_2d = np.zeros((len_yx[0]*(shape_y-2*padding),len_yx[1]*(shape_x-2*padding)))

    for i in range(len(match)):
        sj, ej, sk, ek, sjc, ejc, skc, ekc = match[i]

        if sj != 0:
            sj = sj + padding
        if sk != 0:
            sk = sk + padding
        ej = sj + shape_y-2*padding
        ek = sk + shape_x-2*padding

        data_2d[sj:ej,sk:ek] = subsets[i][padding:-padding,padding:-padding]

    return data_2d