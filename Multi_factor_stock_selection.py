# 导入函数库
from jqdata import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from alphalens import performance
from alphalens import plotting
from alphalens import tears
from alphalens import utils
import time
import datetime


# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    '''选取十个因子（前10个报告分析得出每个系列表现最好的因子），
    等权重——总市值，市盈率TTM，营业利润同比增长率、
    净资产收益率、前1个月涨跌幅、前1个月日均换手率、
    前1个月波动率、户均持股比例变化、机构持股比例变化、
    近1个月净利润上调幅度
    （最新当年净利润预测平均值/1个月前当年净利润预测平均值-1)'''
    set_params(context)
    set_benchmark(g.stock)
    set_option('use_real_price',True)
    run_daily(get_all_factors,time='before_open', reference_security=g.stock)
    g.stocks = get_avaliable_stocks(context)
    #run_monthly()
    
    
    
    
def set_params(context):
    g.M = 30#M天计算一次 
    g.m = 60
    g.n = 5
    g.days = 0
    g.factors = []
    g.num = 5#仓位 
    g.weights = []
    g.stock ='000300.XSHG'
    g.score = 3


def _factors(context):
    '''确定回归系数 '''
    trade_days = get_trade_days(start_date=context.previous_date+ datetime.timedelta(days=-g.m), end_date=context.previous_date)
    price_index = get_price(g.stocks, start_date=context.previous_date+ datetime.timedelta(days=-g.m),
                        end_date=context.previous_date, frequency='1d',fields='close')['close']
    price_return = get_price_return(price_index)
    all_factors = get_factors_data(context, g.stocks, trade_days)
    all_factors = data_preprocess(all_factors)
    IC_score, IC_= get_IC_score(all_factors, price_index)
    factors_based_on_IC, IC_list, IC_score_list = get_factors_based_on_IC(IC_score)
    if factors_based_on_IC == []:
        g.score = 1
        factors_based_on_IC, IC_list, IC_score_list = get_factors_based_on_IC(IC_score)
        g.score = 3
    all_factors = all_factors[factors_based_on_IC]
    all_data = combine_price_factors(price_return, all_factors)
    all_data = all_data.dropna(axis = 0)
    weights = lr(all_data)
    # func = lambda x,y:x*y
    # weights = map(func,weights,IC_score_list)
    # weights = list(weights)
    return factors_based_on_IC, weights
    
    
def get_all_factors(context):
    if g.days %g.M != 0:
        if g.days > 0:
            # if MACD(context) == 0:
            #     g.MACD =1
            #     for stock in list(context.portfolio.positions):
            #         order_target_value(stock,0)  
            # if MACD(context) == 1:
            #     g.MACD =0
            sell_holding(context)
            # if MACD() == 0:
            #     sell_all(context)
            if g.days%(g.M-1) == 0:
                sell_all(context)
    else:
        g.factors, g.weights = _factors(context)
        log.info(g.factors)
        trade_days = get_trade_days(start_date=context.previous_date+ datetime.timedelta(days=-g.n), end_date=context.previous_date)
        stock_list = get_avaliable_stocks(context)
        price_index = get_price(stock_list, start_date=context.previous_date+ datetime.timedelta(days=-g.n),
                    end_date=context.previous_date, frequency='1d',fields='close')['close']
        all_factors = get_factors_data(context, stock_list, trade_days)
        all_factors = all_factors[g.factors]
        all_factors = data_preprocess(all_factors)
        g.stock_list_chosen = select_stock_list(all_factors)
        log.info(g.stock_list_chosen)
        # if g.MACD == 0:
        buy(context)
    g.days += 1
    # log.info(all_factors)
    
def get_avaliable_stocks(context):
    #过滤st股
    stock_list = get_index_stocks(g.stock)
    is_st = get_extras('is_st', stock_list, start_date=context.current_dt.date()+ datetime.timedelta(days=-365), end_date=context.current_dt.date())
    stock_list = list(is_st.columns[~is_st.any().values])
    # #过滤过去一年停牌股票
    # temp_list = []
    # stock_list_new = []
    # price_index = get_price(stock_list, start_date=context.current_dt.date()+ datetime.timedelta(days=-365),
    #                     end_date=datetime.date.today(), frequency='1d')
    # for stock in stock_list:
    #     for i in range(len(price_index['volume'][stock])):
    #         if price_index['volume'][stock][i] == 0 and stock not in temp_list:
    #             temp_list.append(stock)
    # for stock in stock_list:
    #     if stock not in temp_list:
    #         stock_list_new.append(stock)
    return stock_list
    

    
def get_factors_data(context,stock_list_new, trade_days):
    all_factors = pd.DataFrame()
    for day in trade_days:
        q = query(valuation.code,
                  valuation.market_cap,
                  valuation.turnover_ratio,
                  valuation.ps_ratio, 
                  valuation.pb_ratio, 
                  valuation.pe_ratio, 
                  valuation.pcf_ratio,
                  indicator.inc_revenue_year_on_year,#营业收入同比增长率
                  indicator.inc_operation_profit_year_on_year,#营业利润同比增长率
                  indicator.inc_net_profit_to_shareholders_year_on_year,#归属母公司的净利润同比增长率
                  indicator.roe,#净资产收益率
                  indicator.roa,#总资产报酬率
                  indicator.gross_profit_margin,#销售毛利率
                  (balance.total_liability/balance.total_assets).label('ARL'),#资产负债率
                  (balance.fixed_assets/balance.total_assets).label('FACR'),#固定资产比例
                  balance.total_owner_equities,
                  income.basic_eps,#每股收益
                  (income.total_operating_revenue/income.total_operating_cost).label('RC'),#收入成本之比
                  income.net_profit#净利润
                  ).filter(valuation.code.in_(stock_list_new))
        factors = get_fundamentals(q, day)
        factors['date'] = day
        all_factors = pd.concat([all_factors, factors],axis = 0)
    all_factors = all_factors.set_index(['date','code'])
    all_factors = all_factors.fillna(method = 'bfill')
    all_factors = all_factors.fillna(method = 'ffill')
    return all_factors

def get_stock_price(context, stock_list, trade_days):
    all_price = pd.DataFrame()
    for date in trade_days: 
        price = get_price(stocks, start_date = date, end_date = date, fields = 'close')['close']
        all_price = pd.concat([all_price, price], axis = 0)
    
    all_price = pd.DataFrame(all_price)
    return price

def get_price_return(price):
    price = price.T
    price_return = price.pct_change(axis=1)
    return price_return
    
def combine_price_factors(all_price, all_factors):
    #在all_data最后一列添加return 
    all_factors = all_factors.reset_index()
    all_factors['return'] = np.nan
    for i in range(len(all_factors)):
    #每个样本
        stock = all_factors.code[i]
        date = all_factors.date[i]
        if stock in all_price.index and date in all_price.columns:
            all_factors.iloc[i, -1] = all_price.loc[stock, date]
    all_factors = all_factors.set_index(['date','code'])
    return all_factors
   
def lr(all_data):
    all_data = all_data.dropna(axis=0)
    y = all_data[['return']]
    x = pd.DataFrame(all_data.drop('return', axis =1 ))
    lr = LinearRegression()
    lr.fit(x,y)
    coef_1 = lr.coef_
    return coef_1.T

def select_stock_list(factors_data):
    '''
    回归阶段计算预测的出收益率结果，筛选收益率高的股票
    '''
    #特征值时： g.factors_data （300，9）
    #因子权重：[[ 0.00069852 -0.01057227 -0.01943492 -0.00353188  0.00427335 -0.00386444
    #0.04657917 -0.04427861]]
    #进行矩阵运算：预测收益率
    #（m行，n列）*（n行，l列） = （m行，l列）
    #预测收益率
    factors_data = factors_data.reset_index()
    del factors_data['date']
    factors_data = factors_data.groupby('code').mean()
    stock_return = np.dot(factors_data.values,g.weights)
    factors_data['stock_return'] = stock_return
    factors_data = factors_data.reset_index()
    stock_list_chosen = factors_data.sort_values(by = 'stock_return', ascending = False)['code'][:g.num]
    return stock_list_chosen
         
def get_IC_score(all_data,price):
    IC_score = pd.DataFrame()
    IC_ = pd.DataFrame()
    for factor in all_data.columns:
        single_factor_series = all_data[factor]
        factor_return = utils.get_clean_factor_and_forward_returns(single_factor_series, price, max_loss = 0.99)
        IC = performance.factor_information_coefficient(factor_return)
        a = IC.iloc[:, 1]
        IC_ = pd.concat([IC_,IC.iloc[:,1]])
        IC = pd.Series([IC.mean()[1], len(a[a > 0.02])/len(a),performance.factor_returns(factor_return).iloc[:, 1].mean(),
         IC.mean()[1]/IC.std()[1]])
        IC_score = IC_score.append(IC, ignore_index = True)
    IC_score.columns = [['IC_mean','perc_above_0.02','average_return', 'IR']]
    IC_score['factor'] = all_data.columns
    
    return IC_score,IC_
    
def get_factors_based_on_IC(IC_score):
    IC_score['score'] = np.nan
    for row in range(len(IC_score)):
        score = 0
        if IC_score.iloc[row, 0] > 0.03 or IC_score.iloc[row,0] < -0.03:
            score+=1
        if IC_score.iloc[row,1] > 0.5:
            score+=1
        if IC_score.iloc[row,2] > 0.002 or IC_score.iloc[row,2] < -0.002:
            score+=1
        if IC_score.iloc[row,3] > 0.3 or IC_score.iloc[row,3]<-0.3:
            score+=1
        IC_score.iloc[row,5] = score
    factor_list = []
    IC_score_list = []
    IC_list = []
    for row in range(len(IC_score)):
        if IC_score.iloc[row,5]>=g.score:
            factor_list.append(IC_score.iloc[row,4])
            IC_list.append(IC_score.iloc[row,0])
            IC_score_list.append(IC_score.iloc[row,5])
    return factor_list, IC_list, IC_score_list
            

def data_preprocess(all_data):
    for factor in all_data.columns:
        all_data[factor] = all_data[factor].fillna(0)
        all_data[factor] = mad(all_data[factor])
        all_data[factor] = stand(all_data[factor])
    if 'market_cap' in all_data.columns:
        all_data = neutralization(all_data)
    return all_data
    
def stand(factor):
    '''自实现标准化
    '''
    mean = factor.mean()
    std = factor.std()
    return (factor - mean)/std

def mad(factor):
    """3倍中位数去极值
    """
    # 求出因子值的中位数
    med = np.median(factor)

    # 求出因子值与中位数的差值，进行绝对值
    mad = np.median(abs(factor - med))

    # 定义几倍的中位数上下限
    high = med + (3 * 1.4826 * mad)
    low = med - (3 * 1.4826 * mad)

    # 替换上下限以外的值
    factor = np.where(factor > high, high, factor)
    factor = np.where(factor < low, low, factor)
    return factor    
   
def neutralization(all_data):
    market_cap_factor = all_data['market_cap']
    for factor in all_data.columns:
        if factor != 'market_cap':
            x = market_cap_factor.values
            y = all_data[factor]
            #建立回归方程，市值中性化
            lr = LinearRegression()
            #x要求二维，y要求一维
            lr.fit(x.reshape(-1,1), y)
            y_predict = lr.predict(x.reshape(-1,1))
            all_data[factor] = y - y_predict
    return all_data

def MACD():
    hist = attribute_history(g.stock, 80, '1d', 'close', df=True)['close']
    ma5 = hist.rolling(5).mean()
    ma10 = hist.rolling(10).mean()
    ma30 = hist.rolling(30).mean()
    ma60 = hist.rolling(60).mean()
    if ma5[-1] > ma60[-1] and ma5[-2] < ma60[-2]:
        log.info('金叉')
        return 1
    elif ma5[-1] < ma30[-1] and ma5[-2] > ma30[-2]:
        log.info('死叉')
        return 0

def sell_holding(context):
    current_holding = context.portfolio.positions.keys()
    log.info(current_holding)
    for stock in current_holding:
        cost=context.portfolio.positions[stock].avg_cost
        # 获得股票现价
        price=context.portfolio.positions[stock].price
        # 计算收益率
        ret=price/cost-1
        # 如果收益率小于-0.01，即亏损达到1%则卖出股票，幅度可以自己调，一般10%
        if ret < -0.05:
            order_target_value(stock,0)
            print('触发止损')
        
        if ret > 0.1:
            order_target_value(stock,0)
            print('触发止盈')

def buy(context):
    cash_for_each_stock = context.portfolio.available_cash/g.num
    for stock in g.stock_list_chosen:
        # close_array=get_bars(stock, count=10, unit='1d',fields=['close'])['close']
        # current_price = close_array[-1]
        # if current_price < close_array.mean()
        if context.portfolio.available_cash == 0:
            pass
        order_target_value(stock, cash_for_each_stock)
        
def sell_all(context):
    for stock in context.portfolio.positions.keys():
        # cost=context.portfolio.positions[stock].avg_cost
        # # 获得股票现价
        # price=context.portfolio.positions[stock].price
        # # 计算收益率
        # ret=price/cost-1
        # if ret > -0.01:
        order_target_value(stock,0)