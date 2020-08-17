# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:23:13 2020

@author: allen
"""
import torch
import torch.nn as nn
import numpy as np
import new_dataloader
import trading_period
import trading_period_by_test
import trading_period_by_gate_nocheck
import trading_period_by_gate_mean
import matrix_trading
import os 
import pandas as pd
import torch
import torch.utils.data as Data

import matplotlib.pyplot as plt
import time
path_to_image = "C:/Users/Allen/pair_trading DL/negative profit of 2018/"


path_to_average = "C:/Users/Allen/pair_trading DL2/2017/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "C:/Users/Allen/pair_trading DL2/2017/minprice/"
ext_of_minprice = "_min_stock.csv"

path_to_2015compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2015/" 
path_to_2016compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2016/" 
path_to_2017compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2017/" 
path_to_2018compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2018/" 
ext_of_compare = "_table.csv"

path_to_python ="C:/Users/Allen/pair_trading DL2"

path_to_half = "C:/Users/Allen/pair_trading DL2/2016/2016_half/"
path_to_2017half = "C:/Users/Allen/pair_trading DL2/2017_halfmin/"
path_to_2018half = "C:/Users/Allen/pair_trading DL2/2018_halfmin/"
ext_of_half = "_half_min.csv"

path_to_profit = "C:/Users/Allen/pair_trading DL2/New3_ResNet_2018/2017-201710sym/"

max_posion = 5

def test_reward():
    total_reward = 0
    total_num = 0
    total_trade =[0,0,0]
    action_list=[]
    #actions =[[0.5000000000001013, 1.6058835588357105], [1.1231567674441643, 3.009226460170205], [1.6774656461992412, 8.482170812315225], [2.434225143491954, 5.0301795963708305], [3.838786213786223, 7.405844155844149], [50, 100]]
    #actions =[[0.5000000000001013, 1.6058835588357105], [0.8422529644270367, 2.7302766798420457], [1.42874957000333, 3.312693498451958], [1.681668686169194, 8.472736714557769], [2.054041204437417, 4.680031695721116], [3.1352641629535314, 5.810311903246376], [4.378200155159055, 8.429014740108636], [5.632843137254913, 16.43431372549023], [6.8013888888889005, 13.081481481481516], [50,100]]
    #actions = [[0.5000000000002669, 2.500000000000112], [0.7288428324698772, 4.0090056748083995], [1.1218344155846804, 3.0000000000002496], [1.2162849872773496, 7.4631043256997405], [1.4751902346226717, 3.9999999999997113], [1.749999999999973, 3.4999999999998117], [2.086678832116794, 6.2883211678832325], [2.193017888055368, 4.018753606462444], [2.2499999999999822, 7.500000000000021], [2.6328389830508536, 8.9762711864407], [2.980046948356806, 13.515845070422579], [3.2499999999999982, 5.500000000000034], [3.453852327447829, 11.505617977528125], [3.693027210884357, 6.0739795918367605], [4.000000000000004, 12.500000000000034], [4.151949541284411, 10.021788990825703], [4.752819548872187, 15.016917293233117], [4.8633603238866225, 7.977058029689605], [5.7367647058823605, 13.470588235294136], [6.071428571428564, 16.47435897435901], [6.408839779005503, 10.95488029465933], [7.837962962962951, 12.745370370370392], [8.772727272727282, 18.23295454545456], [9.242088607594926, 14.901898734177237], [100,200]]
    actions = [[0.49999999999998446, 3.5000000000000355], [0.5000000000002669, 2.500000000000112], [0.7499999999999689, 3.9999999999997025], [0.8255813953488285, 4.6886304909561005], [0.9999999999999694, 2.9999999999995994], [1.2174744897959144, 7.459183673469383], [1.24999999999997, 2.9999999999996234], [1.4751902346226717, 3.9999999999997113], [1.749999999999973, 3.4999999999998117], [1.7500000000000058, 6.499999999999994], [1.8023648648648616, 8.797297297297295], [1.9999999999999754, 4.030545112781948], [2.2499999999999822, 7.500000000000021], [2.499999999999994, 4.000000000000044], [2.4999999999999973, 6.028455284552839], [2.7500000000000036, 9.00193610842209], [2.980046948356806, 13.515845070422579], [3.2499999999999982, 5.500000000000034], [3.453852327447829, 11.505617977528125], [3.693027210884357, 6.0739795918367605], [4.000000000000004, 12.500000000000034], [4.1462703962704035, 10.038461538461554], [4.500000000000006, 7.499999999999996], [4.752819548872187, 15.016917293233117], [5.0904605263157805, 8.298245614035086], [5.526881720430115, 16.478494623655948], [5.71363636363637, 13.500000000000018], [5.964062500000008, 11.496874999999998], [6.593427835051529, 10.751288659793836], [7.252808988764048, 16.44382022471911], [7.830188679245271, 12.721698113207568], [8.8031914893617, 16.978723404255312], [8.826086956521737, 19.304347826086953], [9.22258064516128, 14.825806451612925], [100,200]]
    #print(actions[0][0])
    #Net = CNN_classsification1()
    #print(Net)
    Net = torch.load('35NewNew2015-201610.pkl')
    Net.eval()
    #print(Net)
    whole_year = new_dataloader.test_data()
    whole_year = torch.FloatTensor(whole_year).cuda()
    #print(whole_year)
    torch_dataset_train = Data.TensorDataset(whole_year)
    whole_test = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = 1024,      # mini batch size
            shuffle = False,               
            )
    for step, (batch_x,) in enumerate(whole_test):
        #print(batch_x)
        output = Net(batch_x)               # cnn output
        _, predicted = torch.max(output, 1)
        action_choose = predicted.cpu().numpy()
        action_choose = action_choose.tolist()
        action_list.append(action_choose)
   # action_choose = predicted.cpu().numpy()
    action_list =sum(action_list, [])

    
    count_test = 0
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2017compare)]
    #print(datelist[167:])
    profit_count = 0
    for date in sorted(datelist[:]): #決定交易要從何時開始
        
        table = pd.read_csv(path_to_2017compare+date+ext_of_compare)
        mindata = pd.read_csv(path_to_average+date+ext_of_average)
        tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
        #halfmin = pd.read_csv(path_to_half+date+ext_of_half)
        #print(tickdata.shape)
        tickdata = tickdata.iloc[166:]
        tickdata.index = np.arange(0,len(tickdata),1)
        os.chdir(path_to_python)    
        num = np.arange(0,len(table),1)
        for pair in num: #看table有幾列配對 依序讀入

            #action_choose = 0
            #spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
            #spread = spread.T.to_numpy()
            #print(spread)
            for i in range(25):
                if action_list[count_test] == i :
                    open, loss = actions[i][0], actions[i][1] 

            profit,opennum,trade_capital,trading  = trading_period_by_gate_mean.pairs( pair ,166,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0.003, 0.01 , 300000000 )
            #print(trading)
            if profit > 0 and opennum == 1 :
                profit_count +=1
                print("有賺錢的pair",profit)
                """
                flag = os.path.isfile(path_to_profit+str(date)+'_profit.csv')
        
                if not flag :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                else :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False)  
                """
                
                
            elif opennum ==1 and profit < 0 :
                
                print("賠錢的pair :",profit)
                """
                flag = os.path.isfile(path_to_profit+str(date)+'_profit.csv')
        
                if not flag :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                else :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False) 
                """
                
                
            total_reward += profit            
            total_num += opennum
            count_test +=1
            total_trade[0] += trading[0]
            total_trade[1] += trading[1]
            total_trade[2] += trading[2]
            
            
            
            #print(count_test)
    print("利潤  and 開倉次數 and 開倉有賺錢的次數/開倉次數:",total_reward ,total_num, profit_count/total_num)
    print("開倉有賺錢次數 :",profit_count)
    print("正常平倉 停損平倉 強迫平倉 :",total_trade[0],total_trade[1],total_trade[2])
    
          
#test()