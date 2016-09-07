# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:12:16 2016

@author: fox
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

Script to convert wikipedia dump data to text format.
/home/fox/.spyder2/.temp.py

writing:
1. Load the optimizer first
2. Get the mean and var of error for each paramaeter combinations specified in optimizer
3. write them to csv file

reading


"""

import csv
import pickle
import mdp
import itertools
import Oger.evaluation

# load the optimizer pickle, saved during the grid search
optimizer_pkl='outputs/optimizer-600dim-100w2v-rmse-bias.pkl'
out_csv='outputs/optimizer-600dim-100w2v-rmse-bias.csv'

with open(optimizer_pkl,'rb') as pkl_file:
    opt=pickle.load(pkl_file)

#opt.parameters return a list of tuples and each tuple has 2 elements
#1. Name of the node
#2. Parameters used for grid search for node

csv_header=[param_tuple[1] for param_tuple in opt.parameters]

param_space=list(itertools.product(*opt.parameter_ranges))
with open(out_csv,'wb+') as csv_file:
    w=csv.writer(csv_file,delimiter=',',)
    w.writerow(['S.No','RMSE']+csv_header)
    paramspace_dimensions=opt.errors.shape
    for param_index_flat,param_val in enumerate(param_space):
        error=opt.errors[mdp.numx.unravel_index(param_index_flat,paramspace_dimensions)]
        row=[param_index_flat+1,error]+list(param_space[param_index_flat])
        w.writerow(row)


























