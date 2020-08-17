# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:27:37 2020

@author: allen
"""

import numpy as np
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
#import mt
import trading_period
import csv
import random
path_to_average = "C:/Users/Allen/pair_trading DL2/2018/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "C:/Users/Allen/pair_trading DL2/2018/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "C:/Users/Allen/pair_trading DL2/newcompare2018/"
ext_of_compare = "_table.csv"
ext_of_check = "_check.csv"

path_to_python ="C:/Users/Allen/pair_trading DL2"
path=os.path.dirname(os.path.abspath(__file__))+'/results/'
#path_to_check ='C:/Users/Allen/pair_trading D2L/check_kmean/'
#path_to_20action ='C:/Users/Allen/pair_trading DL2/20action kmean2018/'
#path_to_20actionwithtax = 'C:/Users/Allen/pair_trading DL/20action kmean2018 with tax/'
path_to_choose2018 = "C:/Users/Allen/pair_trading DL2/gt2018_check_/"
ext_of_groundtruth = "_ground truth.csv"
path_to_choosegt = "C:/Users/Allen/pair_trading DL2/gt2018/"
max_posion = 5
#actions =[[0.5000000000001013, 1.6058835588357105], [0.8422529644270367, 2.7302766798420457], [1.42874957000333, 3.312693498451958], [1.681668686169194, 8.472736714557769], [2.054041204437417, 4.680031695721116], [3.1352641629535314, 5.810311903246376], [4.378200155159055, 8.429014740108636], [5.632843137254913, 16.43431372549023], [6.8013888888889005, 13.081481481481516], [9.748228520815406, 22.99878210806025]]
#actions =[[0.4500000000001013, 1.6058835588357105], [0.8022529644270367, 2.7302766798420457], [1.47874957000333, 3.312693498451958], [1.651668686169194, 8.472736714557769], [2.084041204437417, 4.680031695721116], [3.1352641629535314, 5.810311903246376], [4.278200155159055, 8.429014740108636], [5.68843137254913, 16.43431372549023], [6.713888888889005, 13.081481481481516], [9.69, 22.99878210806025]]
#[[0.5000000000001013, 1.6058835588357105], [0.6489266547405999, 5.516100178890889], [0.8422529644270367, 2.7302766798420457], [1.2193892045453651, 3.516477272726988], [1.539372976155364, 7.9955843391227885], [1.7500000000000067, 2.9999999999999147], [1.8785562632696622, 9.188747346072223], [2.356606317411384, 4.500000000000039], [2.9682769367764976, 5.522039180765841], [3.5842065868263364, 6.853667664670649], [4.688356164383576, 8.828196347031987], [5.094594594594577, 16.315315315315353], [6.4864718614718795, 12.770562770562801], [9.019366197183082, 16.031690140845093], [9.748228520815406, 22.99878210806025]]
actions = [[0.5000000000002669, 2.500000000000112], [0.7288428324698772, 4.0090056748083995], [1.1218344155846804, 3.0000000000002496], [1.2162849872773496, 7.4631043256997405], [1.4751902346226717, 3.9999999999997113], [1.749999999999973, 3.4999999999998117], [2.086678832116794, 6.2883211678832325], [2.193017888055368, 4.018753606462444], [2.2499999999999822, 7.500000000000021], [2.6328389830508536, 8.9762711864407], [2.980046948356806, 13.515845070422579], [3.2499999999999982, 5.500000000000034], [3.453852327447829, 11.505617977528125], [3.693027210884357, 6.0739795918367605], [4.000000000000004, 12.500000000000034], [4.151949541284411, 10.021788990825703], [4.752819548872187, 15.016917293233117], [4.8633603238866225, 7.977058029689605], [5.7367647058823605, 13.470588235294136], [6.071428571428564, 16.47435897435901], [6.408839779005503, 10.95488029465933], [7.837962962962951, 12.745370370370392], [8.772727272727282, 18.23295454545456], [9.242088607594926, 14.901898734177237], [100,200]]
new_actions = []
for i in range(len(actions)):
    if i == len(actions)-1 :
        new_actions.append([actions[i][0]-10,actions[i][1]-10])
        break
    else :
        if abs(actions[i][0]-actions[i+1][0]) >= 0.1 :
            tmp1 ,tmp2 = actions[i][0] - 0.01,actions[i][1] - 0.01
            new_actions.append([tmp1,tmp2])
        elif abs(actions[i][0]-actions[i+1][0]) <= 0.1 and abs(actions[i][0]-actions[i+1][0]) >= 0.05 :
            tmp1 ,tmp2 = actions[i][0] - 0.005,actions[i][1] - 0.005
            new_actions.append([tmp1,tmp2])
        else :
            tmp1 ,tmp2 = actions[i][0] - 0.001,actions[i][1] - 0.001
            new_actions.append([tmp1,tmp2])
print(new_actions)
#print(a)
#actions = [[0.48000000000001013, 1.6058835588357105], [0.6089266547405999, 5.516100178890889], [0.7099999999999054, 2.9999999999997486], [0.9210140679952115, 2.499999999999602], [1.2393892045453651, 3.516477272726988], [1.439372976155364, 7.9955843391227885], [1.7000000000000067, 2.9999999999999147], [1.8311055731762798, 9.058223833257852], [2.306606317411384, 4.500000000000039], [2.5059701492537314, 11.481343283582085],
           #[2.9582769367764976, 5.522039180765841], [3.5563502109704545, 6.585970464135011], [4.080335195530733, 8.15502793296091], [4.406926229508203, 17.016393442622984], [4.955197368421058, 13.006578947368432], [5.802303921568625, 9.791666666666659], [5.936683417085425, 15.46733668341711], [7.0098070739549875, 12.65916398713829], [9.019366197183082, 16.031690140845093], [50,100]]           
os.chdir(path_to_compare)
reward_list=[]
datelist = [f.split('_')[0] for f in os.listdir(path_to_compare)]
print(datelist)
lower_bound = np.arange(0.5,10,0.25)
upper_bound = np.arange(1.5,25,0.5)
"""
def choose_action(lower_bound,upper_bound) :
    action_list=[]
    count = 0
    l , u = 1,0
    while count < 300 :
        l = np.random.choice(lower_bound,1)
        u = np.random.choice(upper_bound,1)        
        if 1.5*l < u :
            w = np.concatenate((l,u),axis = None)
            w = list(w)
            #print(w)
            action_list.append(w)
            count +=1
    return action_list
    
action_list = choose_action(lower_bound, upper_bound)
actions = sorted(action_list, key = lambda s: s[0])
"""

for date in sorted(datelist):
    table = pd.read_csv(path_to_compare+date+ext_of_compare)
    mindata = pd.read_csv(path_to_average+date+ext_of_average)
    tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
    gt = pd.read_csv(path_to_choosegt+date+ext_of_groundtruth,usecols=["action choose"])
    gt = gt.values
    #gt= np.array(gt)
    #gt= gt.ravel()
    
    
    os.chdir(path_to_python)    
    num = np.arange(0,len(table),1)
    #gt = gt.ravel()
    print(gt[0][0])
    for pair in num: #看table有幾列配對 依序讀入
       # print(pair)
        reward = -0.000001
        open_time = 0 
        loss_time = 0
        open_nn =0
        action_choose = 0
        #spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
        #spread = spread.T.to_numpy()
        #print(spread)
        Bth1 = np.ones((5,1))
        #print(tickdata[str(table.stock1[pair])])
        #TickTP1 = tickdata[[str(table.stock1[pair]),str(table.stock2[pair])]]
        #TickTP1 = TickTP1.T.to_numpy()
        #print(TickTP1)
        #choose = int(gt[pair])
        open,loss = new_actions[gt[pair][0]]
        #for open,loss in sorted(new_actions): #對不同開倉門檻做測試
        print(open,loss)
            
            
        spread ,profit ,open_num, rt = trading_period.pairs( pair ,  table , mindata , tickdata , open , loss , max_posion , 0 , 30000000 )
            #print("利潤 :",profit)
            #print("開倉次數 :",open_num)

        
        reward = profit
        open_time = open
        loss_time = loss
        open_nn = open_num
        action_ = gt[pair]


            
            #plotB1 = np.ones((5,len(spread)))*Bth1
        

            
    #print("====================================================================================")
        flag = os.path.isfile('C:/Users/Allen/pair_trading DL2/gt2018_check_/'+str(date)+'_check.csv')
        
        if not flag :
            df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"open":[open_time],"loss":[loss_time],"reward":[reward],"open_num":[open_nn],"action choose":[action_]})
            df.to_csv(path_to_choose2018+str(date)+'_check.csv', mode='w',index=False)
        else :
            df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"open":[open_time],"loss":[loss_time],"reward":[reward],"open_num":[open_nn],"action choose":[action_]})
            df.to_csv(path_to_choose2018+str(date)+'_check.csv', mode='a', header=False,index=False)
            #print(P1)
            #print(C1)
            

