# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:04:55 2020

@author: allen
"""

import numpy as np
import pandas as pd
from integer import num_weight
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from vecm import para_vecm
from scipy.stats import f
from joblib import Parallel, delayed
import multiprocessing
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def VAR_model(y, p):
    k = len(y.T)  # 幾檔股票
    n = len(y)  # 資料長度
    xt = np.ones((n - p, (k * p) + 1))
    for i in range(n - p):
        a = 1
        for j in range(p):
            a = np.hstack((a, y[i + p - j - 1]))
        a = a.reshape([1, (k * p) + 1])
        xt[i] = a
    zt = np.delete(y, np.s_[0:p], axis=0)
    xt = np.mat(xt)
    zt = np.mat(zt)
    beta = (xt.T * xt).I * xt.T * zt  # 計算VAR的參數
    A = zt - xt * beta  # 計算殘差
    sigma = (A.T * A) / (n - p)  # 計算殘差的共變異數矩陣
    return [sigma, beta]


def order_select(y, max_p):
    k = len(y.T)  # 幾檔股票
    n = len(y)  # 資料長度
    bic = np.zeros((max_p, 1))
    for p in range(1, max_p + 1):
        sigma = VAR_model(y, p)[0]
        bic[p - 1] = np.log(np.linalg.det(sigma)) + np.log(n) * p * (k * k) / n
    bic_order = int(np.where(bic == np.min(bic))[0] + 1)  # 因為期數p從1開始，因此需要加1
    return bic_order


def fore_chow(stock1, stock2, stock1_trade, stock2_trade, model):
    if model == 'model1':
        model_name = 'H2'
    elif model == 'model2':
        model_name = 'H1*'
    else:
        model_name = 'H1'
    y = np.vstack([stock1, stock2]).T
    day1 = np.vstack([stock1_trade, stock2_trade]).T
    k = len(y.T)  # 幾檔股票
    n = len(y)  # formation period 資料長度
    y = np.log(y)
    day1 = np.log(day1)
    h = len(day1) - n
    p = order_select(y, 5)  # 計算最佳落後期數
    # ut , A = VAR_model(y , p)                                                                 # 計算VAR殘差共變異數與參數
    at, A = para_vecm(y, model_name, p)
    ut = np.dot(at, at.T) / len(at.T)
    # A = pd.DataFrame(A)
    A = A.T
    phi_0 = np.eye(k)
    A1 = np.delete(A, 0, axis=0).T
    phi = np.hstack((np.zeros([k, 2 * (p - 1)]), phi_0))
    sigma_t = np.dot(np.dot(phi_0, ut), phi_0.T)  # sigma hat
    ut_h = []
    for i in range(1, h + 1):
        lag_mat = day1[len(day1) - i - p - 1:  len(day1) - i, :]
        lag_mat = np.array(lag_mat[::-1])
        if p == 1:
            ut_h.append(lag_mat[0].T - (A[0].T + np.dot(A[1:k * p + 1].T, lag_mat[1:2].T)).T)
        else:
            ut_h.append(lag_mat[0].T - (A[0].T + np.dot(A[1:k * p + 1].T, lag_mat[1:k * p - 1].reshape([k * p, 1]))).T)
    for i in range(h - 1):
        a = phi[:, i * 2:len(phi.T)]
        phi_i = np.dot(A1, a.T)
        sigma_t = sigma_t + np.dot(np.dot(phi_i, ut), phi_i.T)
        phi = np.hstack((phi, phi_i))
    phi = phi[:, ((p - 1) * k):len(phi.T)]
    ut_h = np.array(ut_h).reshape([1, h * 2])
    e_t = np.dot(phi, ut_h.T)
    # 程式防呆，如果 sigma_t inverse 發散，則回傳有結構性斷裂。
    try:
        tau_h = np.dot(np.dot(e_t.T, np.linalg.inv(sigma_t)), e_t) / k
    except:
        return 1
    else:
        if tau_h > float(f.ppf(0.99, k, n - k * p + 1)):  # tau_h > float(chi2.ppf(0.99,k)):
            return 1  # 有結構性斷裂
        else:
            return 0


def fore_chow_jordan(stock1, stock2, model, Flen, give=False, p=0, A=0, ut=0, maxp=5):
    if model == 'model1':
        model_name = 'H2'
    elif model == 'model2':
        model_name = 'H1*'
    else:
        model_name = 'H1'

    day1 = (np.vstack([stock1, stock2]).T)
    day1 = np.log(day1)
    h = len(day1) - Flen
    k = 2  # 幾檔股票
    n = Flen  # formation period 資料長度

    if give == False:
        y = (np.vstack([stock1[0:Flen], stock2[0:Flen]]).T)
        y = np.log(y)
        p = order_select(y, maxp)
        at, A = para_vecm(y, model_name, p)
        ut = np.dot(at, at.T) / len(at.T)

    Remain_A = A.copy()
    Remain_ut = ut.copy()
    Remain_p = p

    A = A.T
    phi_0 = np.eye(k)
    A1 = np.delete(A, 0, axis=0).T
    phi = np.hstack((np.zeros([k, 2 * (p - 1)]), phi_0))
    sigma_t = np.dot(np.dot(phi_0, ut), phi_0.T)  # sigma hat
    ut_h = []

    for i in range(1, h + 1):
        lag_mat = day1[len(day1) - i - p - 1:  len(day1) - i, :]
        lag_mat = np.array(lag_mat[::-1])
        if p == 1:
            ut_h.append(lag_mat[0].T - (A[0].T + np.dot(A[1:k * p + 1].T, lag_mat[1:2].T)).T)
        else:
            ut_h.append(lag_mat[0].T - (A[0].T + np.dot(A[1:k * p + 1].T, lag_mat[1:k * p - 1].reshape([k * p, 1]))).T)

    for i in range(h - 1):
        a = phi[:, i * 2:len(phi.T)]
        phi_i = np.dot(A1, a.T)
        sigma_t = sigma_t + np.dot(np.dot(phi_i, ut), phi_i.T)
        phi = np.hstack((phi, phi_i))
    phi = phi[:, ((p - 1) * k):len(phi.T)]
    ut_h = np.array(ut_h).reshape([1, h * 2])
    e_t = np.dot(phi, ut_h.T)

    # 程式防呆，如果 sigma_t inverse 發散，則回傳有結構性斷裂。
    try:
        tau_h = np.dot(np.dot(e_t.T, np.linalg.inv(sigma_t)), e_t) / k
    except:
        return Remain_p, Remain_A, Remain_ut, 1
    else:
        if tau_h > float(f.ppf(0.99, k, n - k * p + 1)):  # tau_h > float(chi2.ppf(0.99,k)):
            return Remain_p, Remain_A, Remain_ut, 1  # 有結構性斷裂
        else:
            return Remain_p, Remain_A, Remain_ut, 0


def spread_cross_threshold(trigger_spread, threshold, add_num):
    # initialize array
    check = np.zeros(trigger_spread.shape)
    # put on the condition
    check[(trigger_spread - threshold) > 0] = add_num
    check[:, 0] = check[:, 1]
    # Open_trigger_array
    check = check[:, 1:] - check[:, :-1]
    return check


def tax(payoff, rate):
    tax_price = payoff * (1 - rate * (payoff > 0))
    return tax_price


class Trading(object):
    def __init__(self, tick, table, formation_period, trading_period, avg_min_data, trigger_data, open_times,
                 close_times, stop_loss_times, maxi, tax_cost, cost_gate, capital, dump=False):
        self.tick = tick
        self.table = table  # formation period table
        self.formation_period = formation_period  # 建模時間
        self.trading_period = trading_period  # 交易時間
        self.avg_min_data = avg_min_data  # 平均股價
        self.trigger_data = trigger_data  # 每五秒股價
        self.open_times = open_times  # 開倉倍數
        self.close_times = close_times  # 平倉倍數
        self.stop_loss_times = stop_loss_times  # 停損倍數
        self.maxi = maxi  # 最大股票持有張數
        self.tax_cost = tax_cost  # 交易成本
        self.cost_gate = cost_gate  # 交易門檻
        self.capital = capital  # 每組配對最大資金上限
        self.dump = dump

    def check_fore_lag5_timing(self, x, open_timing):
        stock1_name = self.table.stock1.astype('str', copy=False)
        stock2_name = self.table.stock2.astype('str', copy=False)
        avg_min_stock1 = np.array(self.avg_min_data[stock1_name].T)[x, :]
        avg_min_stock2 = np.array(self.avg_min_data[stock2_name].T)[x, :]
        model_type = self.table.model_type[x]
        count = 0
        p, A, ut, _ = fore_chow_jordan(avg_min_stock1[:self.formation_period + 1],
                                       avg_min_stock2[:self.formation_period + 1],
                                       model_type, self.formation_period)
        for i in range(open_timing // (1 + self.tick * 11), self.trading_period):
            p, A, ut, num = fore_chow_jordan(avg_min_stock1[:self.formation_period + i + 1],
                                             avg_min_stock2[:self.formation_period + i + 1],
                                             model_type, self.formation_period, True, p, A, ut)
            if num == 0:
                count = 0
            else:
                count += num
            if count == 5:
                return i * (1 + self.tick * 11)
        return self.trading_period * (1 + self.tick * 11)

    def check_exit_timing(self, check_close, check_stop_loss):
        sec_trading_period = self.trading_period * (1 + self.tick * 11)
        normal_close_timing = np.argmax(check_close != 0)
        stop_loss_timing = np.argmax(check_stop_loss != 0)

        if normal_close_timing == 0 and stop_loss_timing == 0:
            return -4, sec_trading_period - 2
        else:
            if normal_close_timing == 0:
                return -2, stop_loss_timing
            elif stop_loss_timing == 0:
                return 666, normal_close_timing
            else:
                if normal_close_timing < stop_loss_timing:
                    return 666, normal_close_timing
                else:
                    return -2, stop_loss_timing

    def pairs_trading_back_test(self, stock_date, folder_date, use_adf=True, use_fore_lag5=True):
        std = np.array(self.table.stdev)
        self.table = self.table.iloc[(self.open_times + self.close_times) * std > self.cost_gate, :]
        if len(self.table) == 0:
            result = [[0, 0, 0, 0, 0, 0, 0, 0]]
            result = pd.DataFrame(result)
            return result
        # construct spread
        # self.table = self.table[(self.table['stock1'] == '2303') & (self.table['stock2'] == '2308')]
        self.table.reset_index(drop=True, inplace=True)
        stock1_name = self.table.stock1.astype('str', copy=False)
        stock2_name = self.table.stock2.astype('str', copy=False)
        trigger_stock1 = np.array(self.trigger_data[stock1_name].T)
        trigger_stock2 = np.array(self.trigger_data[stock2_name].T)
        avg_min_stock1 = np.array(self.avg_min_data[stock1_name].T)
        avg_min_stock2 = np.array(self.avg_min_data[stock2_name].T)
        w1 = np.expand_dims(np.array(self.table.w1), axis=1)
        w2 = np.expand_dims(np.array(self.table.w2), axis=1)
        trigger_spread = w1 * np.log(trigger_stock1) + w2 * np.log(trigger_stock2)
        avg_min_spread = w1 * np.log(avg_min_stock1) + w2 * np.log(avg_min_stock2)
        if self.tick:
            trigger_stock1 = trigger_stock1[:, :-48]
            trigger_stock2 = trigger_stock2[:, :-48]
            trigger_spread = trigger_spread[:, :-48]

        # threshold
        std = np.array(self.table.stdev)
        mu = np.array(self.table.Emu)
        up_open = np.expand_dims(mu + self.open_times * std, axis=1)
        up_close = np.expand_dims(mu - self.close_times * std, axis=1)
        down_open = np.expand_dims(mu - self.open_times * std, axis=1)
        down_close = np.expand_dims(mu + self.close_times * std, axis=1)
        up_stop_loss = np.expand_dims(mu + self.stop_loss_times * std, axis=1)
        down_stop_loss = np.expand_dims(mu - self.stop_loss_times * std, axis=1)

        # Where_cross_threshold
        check_up_open = spread_cross_threshold(trigger_spread, up_open, 1)
        check_down_open = spread_cross_threshold(trigger_spread, down_open, 3)
        check_up_close = spread_cross_threshold(trigger_spread, up_close, 1)
        check_down_close = spread_cross_threshold(trigger_spread, down_close, 3)
        check_up_stop_loss = spread_cross_threshold(trigger_spread, up_stop_loss, 1)
        check_down_stop_loss = spread_cross_threshold(trigger_spread, down_stop_loss, 3)

        # Combine open trigger array
        check_open = check_up_open + check_down_open
        # 40分鐘後不開倉
        check_open[:, 40 * (1 + self.tick * 11):] = 0
        open_timing = np.argmax(check_open != 0, axis=1)
        int_w = list()
        i = 0
        minute_open_timing = open_timing // (1 + self.tick * 11)
        while i < len(open_timing):
            if open_timing[i] != 0:

                # 檢查開倉時是否有同時突破停損門檻，如果有即不開倉
                if check_up_stop_loss[i, open_timing[i]] > 0 or check_down_stop_loss[i, open_timing[i]] < 0:
                    check_open[i, open_timing[i]] = 0
                    open_timing = np.argmax(check_open != 0, axis=1)
                    minute_open_timing = open_timing // (1 + self.tick * 11)
                    continue
                # 檢查是否同時破上開倉以及下開倉，如果有即不開倉
                if abs(check_open[i, open_timing[i]]) == 4:
                    check_open[i, open_timing[i]] = 0
                    open_timing = np.argmax(check_open != 0, axis=1)
                    minute_open_timing = open_timing // (1 + self.tick * 11)
                    continue

                # 檢查開倉時ADF是否有過，如果沒過即不在此時間點開倉
                w1, w2 = num_weight(self.table.w1.iloc[i], self.table.w2.iloc[i],
                                    trigger_stock1[i, (open_timing[i] + 1)], trigger_stock2[i, (open_timing[i] + 1)],
                                    self.maxi, self.capital)
                adf_spread = w1 * np.log(avg_min_stock1[i, :(self.formation_period + minute_open_timing[i] + 1)]) + \
                             w2 * np.log(avg_min_stock2[i, :(self.formation_period + minute_open_timing[i] + 1)])
                if use_adf:
                    if adfuller(adf_spread, regression='c')[1] > 0.05:
                        check_open[i, open_timing[i]] = 0
                        open_timing = np.argmax(check_open != 0, axis=1)
                        minute_open_timing = open_timing // (1 + self.tick * 11)
                    else:
                        int_w.append([w1, w2])
                        i += 1
                else:
                    int_w.append([w1, w2])
                    i += 1
            else:
                int_w.append([0, 0])
                i += 1

        result = list()
        for i in range(len(open_timing)):
            condition = abs(check_open[i, open_timing[i]])
            long_short = 0
            if condition == 1:
                long_short = -1
                check_up_close[i, :open_timing[i]+1] = 0
                check_up_stop_loss[i, :open_timing[i]+1] = 0
                record, close_timing = self.check_exit_timing(check_up_close[i, :], check_up_stop_loss[i, :])
                if use_fore_lag5:
                    fore_lag5_timing = self.check_fore_lag5_timing(i, open_timing[i] + 1)
                    if fore_lag5_timing < close_timing:
                        close_timing = fore_lag5_timing
                        record = -3
            elif condition == 3:
                long_short = 1
                check_down_close[i, :open_timing[i]+1] = 0
                check_down_stop_loss[i, :open_timing[i]+1] = 0
                record, close_timing = self.check_exit_timing(check_down_close[i, :], check_down_stop_loss[i, :])
                if use_fore_lag5:
                    fore_lag5_timing = self.check_fore_lag5_timing(i, open_timing[i] + 1)
                    if fore_lag5_timing < close_timing:
                        close_timing = fore_lag5_timing
                        record = -3
            elif condition == 0:
                result.append(
                    [stock1_name.iloc[i], stock2_name.iloc[i], int_w[i][0], int_w[i][1], long_short, 0, 0, 0])
                continue
            else:
                print("Error Condition: ", condition)
                continue

            stock1_open_price = trigger_stock1[i, (open_timing[i] + 1)]
            stock2_open_price = trigger_stock2[i, (open_timing[i] + 1)]
            stock1_close_price = trigger_stock1[i, (close_timing + 1)]
            stock2_close_price = trigger_stock2[i, (close_timing + 1)]
            open_s1_payoff = -long_short * stock1_open_price * int_w[i][0]
            open_s2_payoff = -long_short * stock2_open_price * int_w[i][1]
            close_s1_payoff = long_short * stock1_close_price * int_w[i][0]
            close_s2_payoff = long_short * stock2_close_price * int_w[i][1]
            reward = tax(open_s1_payoff, self.tax_cost) + tax(open_s2_payoff, self.tax_cost) + \
                     tax(close_s1_payoff, self.tax_cost) + tax(close_s2_payoff, self.tax_cost)
            result.append([stock1_name.iloc[i], stock2_name.iloc[i], int_w[i][0], int_w[i][1], long_short, reward, 1,
                           record])
            if self.dump:
                plot_spread(self.tick, stock_date, stock1_name[i], stock2_name[i], trigger_spread[i], up_open[i], down_open[i],
                            up_stop_loss[i], down_stop_loss[i], mu[i], self.open_times, self.open_times,
                            self.stop_loss_times, folder_date, open_timing[i] + 1, close_timing + 1, record, reward)
        result = pd.DataFrame(result)
        return result


def plot_spread(tick, stock_date, stock1, stock2, spread, up_open, down_open, up_stop_loss, down_stop_loss, mu,
                up_open_time, down_open_time, stop_loss_time, folder_date, open_timing, close_timing, status, reward):
    plt.figure(figsize=(20, 10))
    plt.plot(spread)
    plt.hlines(up_open, 0, len(spread) - 1, 'b')
    plt.hlines(down_open, 0, len(spread) - 1, 'b')
    plt.hlines(up_stop_loss, 0, len(spread) - 1, 'r')
    plt.hlines(down_stop_loss, 0, len(spread) - 1, 'r')
    plt.hlines(mu, 0, len(spread) - 1, 'g')
    plt.scatter(open_timing, spread[open_timing], color='', edgecolors='b', marker='o', linewidth=3, zorder=99)
    plt.scatter(close_timing, spread[close_timing], color='', edgecolors='r', marker='o', linewidth=3, zorder=99)
    if status == 666:
        status_comment = '正常平倉'
    elif status == -2:
        status_comment = '碰到停損門檻平倉'
    elif status == -3:
        status_comment = '結構性斷裂平倉(fore_lag5)'
    elif status == -4:
        status_comment = '時間結束，強迫平倉'
    else:
        status_comment = 'Error'
    status_comment = status_comment + ' ' + str(open_timing) + ' ' + str(close_timing)
    plt.title(stock_date + ' s1:' + stock1 + ' s2:' + stock2 + ' 上開倉門檻:' + str(up_open_time) + ' 下開倉門檻:'
              + str(down_open_time) + ' 停損門檻:' + str(stop_loss_time))
    plt.xlabel('profit:' + str(reward) + ' ' + str(status_comment))
    if tick:
        file_comment = 'matrix_tick'
    else:
        file_comment = 'matrix'
    if open_timing == close_timing:
        file_comment += 'same_time'
    plt.savefig('dump_data/' + str(folder_date) + '/jpg/' + stock_date + '_' + stock1 + '_' + stock2 + '_' + str(
        up_open_time) + '_' + str(down_open_time) + '_' + str(stop_loss_time) + '_' + file_comment + '.jpg')
    plt.close('all')