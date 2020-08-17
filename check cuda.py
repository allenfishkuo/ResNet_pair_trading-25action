# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:36:00 2020

@author: allen
"""
path_to_average = "C:/Users/Allen/pair_trading DL/2016/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "C:/Users/Allen/pair_trading DL/2016/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "C:/Users/Allen/pair_trading DL/compare/"
ext_of_compare = "_table.csv"
path_to_python ="C:/Users/Allen/pair_trading DL"
path_to_groundtruth = "C:/Users/Allen/pair_trading DL/ground truth trading period/"
ext_of_groundtruth = "_ground truth.csv"
path_to_choose = "C:/Users/Allen/pair_trading DL/6action choose/"


import torch
import pandas as pd 
import os
print(torch.cuda.is_available())
def check_reward():
    datelist = [f.split('_')[0] for f in os.listdir(path_to_compare)]
    #print(datelist)
    count = 0
    for date in sorted(datelist):
        count +=1
        #table = pd.read_csv(path_to_compare+date+ext_of_compare)
        #mindata = pd.read_csv(path_to_average+date+ext_of_average)
#        tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
        gt = pd.read_csv(path_to_choose+date+ext_of_groundtruth,usecols=["reward","action choose"])
        gt = gt.values
        #print(gt)
        #print(gt[pair][0])
        for pair in range(len(gt)):
            if gt[pair][1] == 5 :
                print(gt[pair][0])
                
                
                
#check_reward()