# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:34:32 2020

@author: allen
"""

import numpy as np
import pandas as pd
import os 
from numba import jit
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler 

path_to_average = "C:/Users/Allen/pair_trading DL2/2016/averageprice/"
ext_of_average = "_averagePrice_min.csv"

path_to_minprice = "C:/Users/Allen/pair_trading DL2/2016/minprice/"
ext_of_minprice = "_min_stock.csv"

#path_to_half = "C:/Users/Allen/pair_trading DL/2016/2016_halfmin/"   #2016halfmin data
path_to_2015half = "C:/Users/Allen/pair_trading DL2/2015_halfmin/"
path_to_2016half = "C:/Users/Allen/pair_trading DL2/2016_halfmin/"
path_to_2017half = "C:/Users/Allen/pair_trading DL2/2017_halfmin/"
path_to_2018half = "C:/Users/Allen/pair_trading DL2/2018_halfmin/"
ext_of_half = "_half_min.csv"

#path_to_compare = "C:/Users/Allen/pair_trading DL/compare/"      #2016 halfmin spread w1w2

ext_of_compare = "_table.csv"
path_to_2015compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2015/" 
path_to_2016compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2016/" 
path_to_2017compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2017/" 
path_to_2018compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2018/" 

path_to_python ="C:/Users/Allen/pair_trading DL2"
path_to_groundtruth = "C:/Users/Allen/pair_trading DL2/ground truth trading period/"
ext_of_groundtruth = "_ground truth.csv"
path_to_choose2015 = "C:/Users/Allen/pair_trading DL2/newgt2015/"
path_to_choose2016 = "C:/Users/Allen/pair_trading DL2/newgt2016/"
path_to_choose2017 = "C:/Users/Allen/pair_trading DL2/newgt2017/"
path_to_choose2018 = "C:/Users/Allen/pair_trading DL2/newgt2018/"
min_max_scaler = preprocessing.MinMaxScaler()

no_half =["2231","8454","6285","2313","2867","1702","3662","1536","9938","2847","6456"]

SS = StandardScaler()

#@jit
def read_data():
    number_of_kmean = 25
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    count_number =[0]*number_of_kmean
    count_test =[0]*number_of_kmean


    dic_compare = { 0 : path_to_2015compare, 1 : path_to_2016compare, 2 : path_to_2017compare ,3:path_to_2018compare}
    dic_half = { 0 :path_to_2015half, 1 :path_to_2016half, 2 :path_to_2017half,3:path_to_2018half}
    dic_choose = { 0 : path_to_choose2015, 1 :path_to_choose2016, 2 :path_to_choose2017,3:path_to_choose2018}
    for year in range(1,len(dic_choose)-2):
        
        datelist = [f.split('_')[0] for f in os.listdir(dic_compare[year])] #選擇幾年度的
        #print(dic_compare[year])
        count = 0
        for date in sorted(datelist):
            #print(date)
            count +=1
            table = pd.read_csv(dic_compare[year]+date+ext_of_compare)
            halfmin = pd.read_csv(dic_half[year]+date+ext_of_half)
            gt = pd.read_csv(dic_choose[year]+date+ext_of_groundtruth,usecols=["action choose"])
            gt = gt.values
            #print(date)
            #print(count)

            for pair in range(len(table)):
                if (str(table.stock1[pair]) in no_half or str(table.stock2[pair]) in no_half) and year == 0 :
                    continue
                else :                    
                    spread = table.w1[pair] * np.log((halfmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(halfmin[ str(table.stock2[pair]) ])
                    spread = spread[32:332].values
                    spread = preprocessing.scale(spread)
                    new_spread = np.zeros((3,512))
                    new_spread[0,106:406] = spread 
                    mindata1 = halfmin[str(table.stock1[pair])][32:332].values
                    mindata2 = halfmin[str(table.stock2[pair])][32:332].values
                    mindata1 = preprocessing.scale(mindata1)
                    mindata2 = preprocessing.scale(mindata2)       
                    new_spread[1,106:406] = mindata1
                    new_spread[2,106:406] = mindata2
                    
                    if date[0:4] != "2016" :
                    
                        number = gt[pair][0]
                        for i in range(number_of_kmean): #幾個action
                            if number == i : 
                                train_data.append(new_spread)
                                count_number[number] +=1
                                train_label.append(gt[pair])
                    
            
                    else:                    
                        if date[0:6] != "201611" and date[0:6] !="201612" : #9月以前
                            number = gt[pair][0]
                            for i in range(number_of_kmean): #幾個action
                                if number == i : 
                                    train_data.append(new_spread)
                                    count_number[number] +=1
                                    train_label.append(gt[pair])
            
                        else: 
                            number = gt[pair][0]
                            for i in range(number_of_kmean): #幾個action
                                if number == i  :
                                    test_data.append(new_spread)
                                    count_test[number] +=1
                                    test_label.append(gt[pair])    
                                
    
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    
    
    #print(train_data)
    #train_data = preprocessing.scale(train_data,axis=1)
    #test_data = preprocessing.scale(test_data,axis=1)
    #train_data = preprocessing.normalize(train_data,norm ="l1")
    """
    for i in range(train_data.shape[1]):
            train_data[:, i, :] = preprocessing.scale(train_data[:, i, :],axis=1)
    print(train_data)
    
    for i in range(test_data.shape[1]):
            test_data[:, i, :] = preprocessing.scale(test_data[:, i, :],axis=1)
    #test_data = preprocessing.scale(test_data,axis=1)
    #test_data = preprocessing.normalize(test_data,norm ="l1")
    print(train_data.shape)
    """
    #print(train_data)
  
    train_label = np.asarray(train_label)
    test_label = np.asarray(test_label)
    train_label = train_label.flatten()
    test_label = test_label.flatten()
    print(train_label.shape)
    print(count_number)
    print(count_test)
    print(train_data.shape)
    
    print(test_data.shape)
    
    return train_data, train_label, test_data, test_label

def test_data():
    whole_year = []
    #test_data = []

    #test_label = []
    #counts = [0,0,0,0,0]
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2017compare)]
    #print(datelist)
    count = 0
    for date in sorted(datelist[:]): #10月開始
        count +=1
        table = pd.read_csv(path_to_2017compare+date+ext_of_compare)
        #mindata = pd.read_csv(path_to_average+date+ext_of_average)
        halfmin = pd.read_csv(path_to_2017half+date+ext_of_half)
        
#        tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
        #print(date)
        #print(count)
        for pair in range(len(table)):
            spread = table.w1[pair] * np.log((halfmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(halfmin[ str(table.stock2[pair]) ])
            spread = spread[32:332].values
            spread = preprocessing.scale(spread)
            new_spread = np.zeros((3,512))
            new_spread[0,106:406] = spread  
            
            mindata1 = halfmin[str(table.stock1[pair])][32:332].values
            mindata2 = halfmin[str(table.stock2[pair])][32:332].values
            mindata1 = preprocessing.scale(mindata1)
            mindata2 = preprocessing.scale(mindata2)
            
                  
            new_spread[1,106:406] = mindata1
            new_spread[2,106:406] = mindata2
            """
            plt.figure()
            for i in range(new_spread.shape[0]):
                plt.plot(new_spread[i,:])
            plt.show()
            plt.close()
            """

            whole_year.append(new_spread)
                    #count_number[number] +=1
                    #train_label.append(gt[pair])

    whole_year = np.asarray(whole_year)
    #whole_year = preprocessing.scale(whole_year,axis=1)
    #train_data = preprocessing.normalize(train_data,norm ="l1")
    #test_data = np.asarray(test_data)
    #test_data = preprocessing.scale(test_data,axis=1)
    #test_data = preprocessing.normalize(test_data,norm ="l1")
    #print(whole_year)
    print("whole_year :",whole_year.shape)
    #print(test_data)

    #print(test_data.shape)
    return whole_year
    
#read_data()
#test_data()
