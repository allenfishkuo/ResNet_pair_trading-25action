# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:48:00 2020

@author: allen
"""
import numpy as np
import time
import sys
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
ext_of_check = "_check.csv"
ext_of_groundtruth  = "_ground truth.csv"

path_to_python ="C:/Users/Allen/pair_trading DL2"
path=os.path.dirname(os.path.abspath(__file__))+'/results/'
path_to_check ='C:/Users/Allen/pair_trading DL2/gt2018_check_/'
path_to_check1 ='C:/Users/Allen/pair_trading DL2/gt2018/'

def check_equal():
    datelist = [f.split('_')[0] for f in os.listdir(path_to_check)]
    reward_index = []
    dif_reward = 0
    total_gt = 0
    dif_reward_index = []
    count = 0
    for date in sorted(datelist):
        
        gt = pd.read_csv(path_to_check1+date+ext_of_groundtruth,usecols=["reward"])
        check = pd.read_csv(path_to_check+date+ext_of_check,usecols=["reward"])
        gt = gt.values
        check = check.values        

        if date[0:6] == "201801":
            total_gt += len(gt)
            for i in range(gt.shape[0]): 
                if check[i,0] == 0 and gt[i,0] == 0 :
                    continue
                
                dif = (check[i,0] - gt[i,0])/gt[i,0]
                if check[i,0] != gt[i,0]:
                    dif_reward_index.append([check[i,0],gt[i,0]])
                reward_index.append(dif)                    
                count+=1
                if dif != 0 :
                    dif_reward += 1 
                
    num = np.arange(1,count+1,1)
    #print(reward_index)        
    reward_index = np.array(reward_index)
    #print(len(num),len(reward_index))
    #print(reward_index)  
    plt.figure(figsize=(16, 12))
    plt.bar(num,reward_index)
    plt.xlabel("201801 stock pairs")
    plt.ylabel("reward error")
    plt.xlim(1,len(num)+1)
    #plt.ylim(0, 4) 
    print("不同reward 數",dif_reward)
    print(dif_reward_index)
    
check_equal()