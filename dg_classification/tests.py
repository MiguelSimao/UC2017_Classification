#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 19:33:16 2018

@author: simao
"""


def tsunroll(X,T):
    x = []
    t = []
    sz = np.empty((0,1),int)
    for i,sample in enumerate(X):
        ind_include = ~np.all(sample==0,axis=1)
        sample = sample[ind_include]
        x.append(sample)
        t.append(T[i][ind_include])
        if np
        sz = np.append(sz,np.argwhere(~ind_include)[0]-1)
    return np.concatenate(x), np.concatenate(t), sz

F_val,T_val,sz_val = tsunroll(F_val,T_val)