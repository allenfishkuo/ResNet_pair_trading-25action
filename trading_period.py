# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:08:05 2018

@author: chuchu0936
"""

# from formation_period import formation_period
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
# from Predict_Client import send_request
from cost import tax, slip
from integer import num_weight
from MTSA import fore_chow, spread_chow
# import tensorflow
# from keras.models import load_model
import pandas as pd
import numpy as np

# 標準差倍數當作停損門檻(滑價＋交易稅)-------------------------------------------------------------------------------

def pairs( pair ,  table , min_data , tick_data , open_time , stop_loss_time , maxi , tax_cost , capital ):
    
    #table = pd.DataFrame(table).T
    #print(table)
    #min_price = day1
    #min_price = min_price.dropna(axis = 1)
    #min_price.index  = np.arange(0,len(min_price),1)
    
    #num = np.arange(0,len(table),1)
    
    #t = formate_time                                           # formate time
    #print(table)
    #print(min_data)
    #print(tick_data)
    local_open_num = []
    local_profit = []
    local_rt = []
    #for pair in num:
    
    spread = table.w1[pair] * np.log((min_data[ str(table.stock1[pair] )])) + table.w2[pair] * np.log(min_data[ str(table.stock2[pair]) ])
    #print(spread)
    up_open = table.Emu[pair] + table.stdev[pair] * open_time                      # 上開倉門檻
    down_open = table.Emu[pair] - table.stdev[pair] * open_time                    # 下開倉門檻
        
    stop_loss = table.stdev[pair] * stop_loss_time                                # 停損門檻
    #print(up_open,down_open,stop_loss)  
    close = table.Emu[pair]                                                        # 平倉(均值)
        
    M = round( table.zcr[pair] * len(spread) )                                    # 平均持有時間
        
    trade = 0                                                                     # 計算開倉次數
    #discount = 1
    
    position = 0                                                                  # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2：強制平倉
    
    #model=load_model('model.h5')
    #model.summary()
    
    pos = []
    stock1_profit = []
    stock2_profit = []
    stop_trade = False
    for i in range( len(spread)-2 ):
            
        #stock1_seq = min_price[ table.stock1[pair] ].loc[0:t+i]
        #stock2_seq = min_price[ table.stock2[pair] ].loc[0:t+i]
            
        if position == 0 and len(spread)-i > M and i>165 and i <265:                                   # 之前無開倉
        
            if ( spread[i] - up_open ) * ( spread[i+1] - up_open ) < 0 :
                    
                # 資金權重轉股票張數，並整數化
                w1 , w2 = num_weight( table.w1[pair] , table.w2[pair] , tick_data[str(table.stock1[pair])][(i+2)] , tick_data[str(table.stock2[pair])][(i+2)] , maxi , capital )          
                    
                #print("整數畫",w1,w2)
                        
                position = -1
                    
                stock1_payoff = w1 * slip( tick_data[str(table.stock1[pair])][(i+2)] , table.w1[pair] )
                    
                stock2_payoff = w2 * slip( tick_data[str(table.stock2[pair])][(i+2)] , table.w2[pair] )
                        
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )  # 計算交易成本
                #print("上開倉",stock1_payoff, stock2_payoff)
                trade = trade + 1
                
            elif ( spread[i] - down_open ) * ( spread[i+1] - down_open ) < 0 :
                    
                # 資金權重轉股票張數，並整數化
                w1 , w2 = num_weight( table.w1[pair] , table.w2[pair] , tick_data[str(table.stock1[pair])][(i+2)] , tick_data[str(table.stock2[pair])][(i+2)] , maxi , capital )          
                    
                #spread1 = w1 * np.log(stock1_seq) + w2 * np.log(stock2_seq)
                    
               # if adfuller( spread1 , regression='ct' )[1] > 0.05 :                                    # spread平穩才開倉
                        
                 #   position = 0
                        
                 #   stock1_payoff = 0
                        
                 #   stock2_payoff = 0

                 #   stop_trade = True
                        
                #else:
                    
                position = 1
                    
                stock1_payoff = -w1 * slip( tick_data[str(table.stock1[pair])][(i+2)] , -table.w1[pair] )
                    
                stock2_payoff = -w2 * slip( tick_data[str(table.stock2[pair])][(i+2)] , -table.w2[pair] )
                        
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )   # 計算交易成本
                #print("下開倉",stock1_payoff, stock2_payoff)
                trade = trade + 1
                    
            else: 
                    
                position = 0
        
                stock1_payoff = 0
            
                stock2_payoff = 0
        
        elif position == -1:                                                                         # 之前有開空倉，平空倉
            
            #spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            
            #temp=spread1[i+1:t+i+1].reshape(1,150,1)
            #pre=model.predict_classes(temp)
            
            if ( spread[i] - down_open) * ( spread[i+1] - down_open ) < 0 :

                position = 0                                                                         # 平倉
            
                stock1_payoff = -w1 * slip( tick_data[str(table.stock1[pair])][(i+2)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[str(table.stock2[pair])][(i+2)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )       # 計算交易成本

                stop_trade = True
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif ( spread[i] - (close + stop_loss) ) * ( spread[i+1] - (close + stop_loss) ) < 0 :
                
                position = -2                                                                                   # 碰到停損門檻，強制平倉
            
                stock1_payoff = -w1 * slip( tick_data[str(table.stock1[pair])][(i+2)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[str(table.stock2[pair])][(i+2)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )       # 計算交易成本

                stop_trade = True
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif adfuller( spread1 , regression='ct' )[1] > 0.05 :
                
                #position = -3                                                                                    # 出現單跟，強制平倉
            
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif fore_chow( min_price[table.stock1[pair]].loc[0:t] , min_price[table.stock2[pair]].loc[0:t] , stock1_seq , stock2_seq ) == 1 :
                
                #position = -3                                                                                    # 結構性斷裂，強制平倉
                
                #stock1_payoff = -w1 * slip( tick_data[str(table.stock1[pair])][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[str(table.stock2[pair])][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本

               # stop_trade = True
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif spread_chow( spread1 , i ) == 1 :
                
                #position = -2                                                                                    # 結構性斷裂，強制平倉
            
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif send_request( spread1 , 150 , 0.9 ) == 1:
                
                #position = -2                                                                                    # LSTM偵測，強制平倉
                
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif ( (pre!=0) | (pre!=4) ):
                
                #position = -2                                                                                    # CNN偵測，強制平倉
                
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif i == (len(spread)-3) :                                                                          # 回測結束，強制平倉
            
                position = -2
            
                stock1_payoff = -w1 * slip( tick_data[str(table.stock1[pair])][len(tick_data)-1] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[str(table.stock2[pair])][len(tick_data)-1] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )         # 計算交易成本

                stop_trade = True
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            else: 
            
                position = -1
        
                stock1_payoff = 0
            
                stock2_payoff = 0
        
        elif position == 1:                                                                                        # 之前有開多倉，平多倉
            
            #spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            
            #temp=spread1[i+1:t+i+1].reshape(1,150,1)
            #pre=model.predict_classes(temp)
            
            if ( spread[i] - up_open ) * ( spread[i+1] - up_open ) < 0 :
        
                position = 0                                                                                       # 平倉
            
                stock1_payoff = w1 * slip( tick_data[str(table.stock1[pair])][(i+2)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[str(table.stock2[pair])][(i+2)] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )           # 計算交易成本

                stop_trade = True
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif ( spread[i] - (close - stop_loss) ) * ( spread[i+1] - (close - stop_loss) ) < 0 :
                
                position = -2                                                                                       # 碰到停損門檻，強制平倉
            
                stock1_payoff = w1 * slip( tick_data[str(table.stock1[pair])][(i+2)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[str(table.stock2[pair])][(i+2)] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )           # 計算交易成本

                stop_trade = True
                
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif i == (len(spread)-3) :                                                                              # 回測結束，強制平倉
            
                position = -2
            
                stock1_payoff = w1 * slip( tick_data[str(table.stock1[pair])][len(tick_data)-1] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[str(table.stock2[pair])][len(tick_data)-1] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )            # 計算交易成本

                stop_trade = True
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            else: 
            
                position = 1
        
                stock1_payoff = 0
                    
                stock2_payoff = 0
            
        else:
                
            if position == -2 or position == -3:
                
                stock1_payoff = 0
                
                stock2_payoff = 0
                    
            else:
        
                position = 0                                                                         # 剩下時間少於預期開倉時間，則不開倉，避免損失
        
                stock1_payoff = 0
                
                stock2_payoff = 0
                    
        pos.append(position)
            
        stock1_profit.append(stock1_payoff)
        #(stock1_profit)
        stock2_profit.append(stock2_payoff)
        #print(stock2_profit)
        if stop_trade:
            break
    #print(position)
    
    #x = np.arange(0,121)
    #plt.plot(spread)
    #plt.axhline(y=close,color='r')
    #plt.axhline(y=up_open,color='r')
    #plt.axhline(y=down_open,color='r')
    #plt.axhline(y=close+stop_loss,color='green')
    #plt.axhline(y=close-stop_loss,color='green')
            
    #bp = np.array(np.where( pos == -3 ))
            
    #if bp.size != 0:
                
        #plt.axvline(x=bp[0][0],color='green')
            
    #plt.show()
    
    trading_profit = sum(stock1_profit) + sum(stock2_profit)
    #print(trading_profit)   
    """
    if 1.8 * table.stdev[pair] < tax_cost:
        
        trading_profit = 0
            
        trade = 0
    """   
    #local_profit.append(trading_profit)
    #local_profit = trading_profit
    
    local_open_num.append(trade)
    #local_open_num = trade
        
    if trade == 0:            # 如果都沒有開倉，則報酬為0
        
        local_rt.append(0)
        #local_rt = 0
        
    else:                     # 計算平均報酬
        
        local_rt.append(trading_profit/(capital*trade) )
        #local_rt = trading_profit/(capital*trade)
        
    #posi = pos[len(spread)-2]
    '''
    if tax_cost == 0:
    
        local_profit = pd.DataFrame(local_profit)       ; local_profit.columns = ["profit without cost"]
        
    else:
        
        local_profit = pd.DataFrame(local_profit)       ; local_profit.columns = ["profit"]
        
    local_open_num = pd.DataFrame(local_open_num)   ; local_open_num.columns = ["open number"]
    local_rt = pd.DataFrame(local_rt)               ; local_rt.columns = ["return"]
    
    #back_test = pd.concat([local_profit,local_open_num,local_rt],axis=1)
    '''
    return  spread , trading_profit , local_open_num , local_rt #, 0
    
    