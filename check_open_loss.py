# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:12:31 2020

@author: Allen
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:13:25 2020

@author: Allen
"""

import numpy as np
import pandas as pd
import os 
import collections
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler 
path_to_average = "C:/Users/Allen/pair_trading DL/2016/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "C:/Users/Allen/pair_trading DL/2016/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "C:/Users/Allen/pair_trading DL/compare/"
ext_of_compare = "_table.csv"
path_to_python ="C:/Users/Allen/pair_trading DL"
path_to_groundtruth = "C:/Users/Allen/pair_trading DL/ground truth trading period/"
ext_of_groundtruth = "_ground truth.csv"
path_to_choose = "C:/Users/Allen/pair_trading DL/action_choose/"
path_to_all = "C:/Users/Allen/pair_trading DL/all answer for new/"

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer
path_to_image ='C:/Users/Allen/pair_trading DL2/profit image/'
min_max_scaler = preprocessing.MinMaxScaler()

SS = StandardScaler()
def check_distributed():

    train_label = []
    test_label = []

    gt_date = [f.split('_')[0] for f in os.listdir(path_to_all)]
   # print(gt_date)
    count2 = 0
    for date in sorted(gt_date):
        count2 +=1
        gt = pd.read_csv(path_to_all+date+ext_of_groundtruth,usecols=["open","loss"])
        #print(gt.iloc[0,0])
        gt = gt.values
        #print(gt)
        for l in range(len(gt)):
            if count2 <=186 :
                #number = gt[l][0]
                #print(number)
                #count_number[number] +=1
                train_label.append(gt[l])
            else :
                #number = gt[l][0]
                #count_test[number] +=1
                test_label.append(gt[l])
        
    #print(train_label)
   # print(test_label.shape)
    
    train_label = np.asarray(train_label)
    #print(train_label[:,1])
    
    
    #check_ol = [[3.75, 7.5]]
    #print(check_ol[0,0]
    plt.figure(figsize=(10, 5))
    unique, counts = np.unique(train_label,axis = 0, return_counts=True)
    my_labels={"1":'# of Trades:>500',"2":'# of Trades:500~300',"3":'# of Trades:<300',"4":'Kmeans centers'}
    for i in range(len(unique)):
        if counts[i] >= 500 :
            plt.scatter(unique[i,0],unique[i,1], c ="red",alpha=0.4)
                        
        elif counts[i] < 500 and counts[i] > 300:
            plt.scatter(unique[i,0],unique[i,1], c ="yellow",alpha=0.4)
        else :
            plt.scatter(unique[i,0],unique[i,1], c ="blue",alpha=0.4)
    print(len(unique))
    #model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors',assign_labels='kmeans')
    #labels = model.fit_predict(train_label)
    #plt.scatter(train_label[:, 0], train_label[:, 1], c=labels,s=50, cmap='viridis')
    kmeans = KMeans(n_clusters=35)
    kmeans.fit(train_label)
    centers = kmeans.cluster_centers_ 
    plt.scatter(unique[0,0],unique[0,1], c ="red",alpha=0.4,label=my_labels["1"])
    plt.scatter(unique[0,0],unique[0,1], c ="yellow",alpha=0.4,label=my_labels["2"])
    plt.scatter(unique[0,0],unique[0,1], c ="blue",alpha=0.4,label=my_labels["3"])
    plt.scatter(centers[:, 0], centers[:, 1], c='black',alpha=0.7,label=my_labels["4"])
    means=[]
    
    for i in range(len(centers)):
        means.append([centers[i,0],centers[i,1]])
    #plt.figure()
    #plt.title("Ground truth distributed")
    plt.xlabel("Opening Trigger")
    plt.ylabel("Stop-Loss Trigger")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(path_to_image+"Pairs trading Kmeans.png")
    plt.show()
    plt.close()
    print(sorted(means))
    plt.figure()
    visualizer = KElbowVisualizer(kmeans, k=(15,40))

    visualizer.fit(train_label)        # Fit the data to the visualizer
    visualizer.show()
    

    
    
    
check_distributed()