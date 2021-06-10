#This is ManualStrategy implementation
import numpy as np
import pandas as pd
import datetime as dt
import indicators as ind
import marketsimcode as mk
import matplotlib.pyplot as plt

from util import get_data


def testPolicy(symbol = "JPM", sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), sv = 100000):
    """ManualStrategy is man made policy using three indicators

        :symbol: stock symbol to apply policy
        :type symbol: str
        :sd: start date
        :type sd: datetime
        :ed: end date
        :type ed: datetime
        :sv: start value
        :type sv: int
        :retrun trade dataframe
        :rtype: dataframe

    """

    #-------------------------------------------Read in date----------------------------------------------
    # start_date = dt.datetime.strftime(sd, '%Y-%m-%d')
    # end_date = dt.datetime.strftime(ed, '%Y-%m-%d')
    dates = pd.date_range(sd, ed)
    price = get_data([symbol], dates)
    price = price[symbol] # remove spy

    #-----------------------------------------Compute Indicators------------------------------------------
    """ Selected 3 indicators: bbp, cci, rsi"""
    #bbp compute
    sma, top_band, bottom_band, bbp = ind.bbp([symbol], sd, ed, period=20)
    #macd compute
    # ema_12, ema_26, macd, signal, macd_hist = ind.macd([symbol], sd, ed)
    #rsi compute
    rsi = ind.rsi([symbol], sd, ed, period= 14)
    #cci compute
    cci_val, tp, sma_tp = ind.cci([symbol], sd, ed, 20)
    # print(bbp, macd, rsi,macd_hist)

    #-----------------------------------------Create signal based indicators------------------------------
    """set different weight to to indicators
        bbp: rsi: cci = 1:0.8:0.6  
        set sum(signals) >= 0.6 --> Buy 1000
        set sum(signals) <= -0.6 --> short 1000

    """
    df = pd.DataFrame()
    df['price'] = price
    df['bbp_signal'] =  [-1 if x > 1 else 1 if x < 0 else 0 for x in bbp.values] # df.values convert df to ndarray format
    df['rsi_signal'] = [-0.8 if x > 70 else 0.8 if x < 30 else 0 for x in rsi.values]

    # macd_df = pd.DataFrame()
    # macd_df['diff'] = macd_hist.iloc[:,0]
    # macd_df['cross'] = [1 if x > 0 else 0 for x in macd.values]
    # macd_df['cross_diff'] = macd_df['cross'].diff()
    # macd_df[0,'cross_diff'] = 0 #change first nan to 0 after .diff() 
    # macd_df['signal'] = [0.6 if ((x[0] > 0) & (x[2] == 1)) else -0.6 if ((x[0] < 0) & (x[2] == -1)) else 0 for x in macd_df.values]

    # df['macd_signal'] = macd_df['signal']

    df['cci'] = [-0.6 if x > 1 else 0.6 if x < 0 else 0 for x in cci_val.values]
    
    #------------------------------------------------------holding frame-------------------------------------------
    holding = price.copy()
    threshold = 0.8
    holding[df.iloc[:,1:].sum(axis = 1) >= threshold] = 1000
    holding[df.iloc[:,1:].sum(axis=1) <= -threshold] = -1000
    holding[((df.iloc[:,1:].sum(axis = 1) < threshold) & (df.iloc[:,1:].sum(axis = 1) > -threshold))] = float("nan") #set var as nan in order to use .fillna()

    #fill forward in holding
    holding.fillna(method = "ffill", inplace = True) #after fillna transferring to pd.Series 
   
    #update df_trades each day
    df_trades = price.copy()
    df_trades = holding.diff()
    #set inital day trading is holding status
    df_trades.iloc[0] = holding.iloc[0]  # .diff() --> first element would be Nan

    df_trades= df_trades.to_frame() #convert to pd.DataFrame

    return df_trades

#implement author func
def author(self):
    return 'yjiang347'
        
if __name__ == "__main__":

    df_trades = testPolicy()
    
    # I. get manual strategy portvals
    port_val = mk.compute_portvals(orders_file = df_trades, start_val=100000, commission=9.95, impact=0.005)
    port_val = port_val/port_val.iloc[0]	 

    ## II. create bench trades df
    price = get_data(["JPM"] ,pd.date_range(dt.datetime(2008,1,1),dt.datetime(2009, 12, 31)))
    price = price.iloc[:,1:]#remove spy
    bench_trades = price.copy()
    bench_trades.iloc[0,:] = 1000
    bench_trades.iloc[1:,:] = 0
    ## III. get bench daily portfolio value 
    bench_portval = mk.compute_portvals(orders_file = bench_trades, start_val=100000, commission=9.95, impact=0.005)
    bench_portval = bench_portval/bench_portval.iloc[0,:]

    #---------------------------Plot benchmark vs, Manual Strategy----------------------
    # fig = plt.figure(figsize = (12,8))
    # plt.plot(port_val, label = "ManualStrategy", color = 'red')
    # plt.plot(bench_portval, label = "Benchmark", color = 'green')
    # plt.title('Benchmark vs. Manual Strategy\n(Portfolio Value)')
    # plt.ylabel('Normalized Portfolio Value')
    # plt.xlabel('Date')
    # plt.legend(loc = 'upper left')
    # plt.xticks(rotation = 30)
    # plt.savefig('portval.png')
   