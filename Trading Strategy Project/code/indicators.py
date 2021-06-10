"""
Part 1 of PR6
This function implement 5 technical indicatiors:
(1) Bollinger Band %b(BBP) -- Volatility (-1~1)
(2) Relative Strength Index (RSI) --  Momentum (0-100)
(3) Moving average converge/diverge(MACD) -- Trends
(4) On Balance Volume (OBV) -- Volume
(5) Commodity Channel Index (CCI) -- Others
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

#eliminate future warning for marplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from util import get_data

#--------------------------------------- RSI - Momentum --------------------------------------------
def rsi(syms, start, end, period= 14):
    """This function calculates rsi indicator 

    :param syms: symbols
    :type syms: list
    :param start: start date
    :type start: str
    :param end: end date
    :type end: str
    :param period : time-window
    :type period: int
    :return: momentum dataframe
    :rtype: pd.dataframe
    """
    #read in data
    df = get_data(syms, pd.date_range(start-dt.timedelta(days = period*3), end))
    #normalize price
    df = df/df.iloc[0,:]
    #remove spy
    df = df.iloc[:,1:]
    #calculate daily return
    daily_rets = df/df.shift(1) - 1
    daily_rets.iloc[0,:] = 0
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum() #get abs value
    #avg gain within each period(14-day)
    avg_gain = daily_rets.copy()
    avg_gain.iloc[:,:] = 0
    avg_gain.values[period:, :] = (up_rets.values[period:, :] - up_rets.values[:-period, :])/14
    #avg loss within each period(14-day)
    avg_loss = daily_rets.copy()
    avg_loss.iloc[:,:] = 0
    avg_loss.values[period:,:] = (down_rets.values[period:,:] - down_rets.values[:-period, :])/14
    #rs and rsi compute
    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    #handle 1) first 14 days and 2)avg_loss = 0 in rs
    rsi.iloc[:14, :] = np.nan
    rsi[rsi == np.inf] = 100

    rsi = rsi.loc[start:,:]
    
    return rsi
   
#--------------------------------------- BBP - Violatility -----------------------------------------
def bbp(syms, start, end, period=20):
    """This function calculates bbp indicator 

    :param syms: symbols
    :type syms: list
    :param start: start date
    :type start: datetime
    :param end: end date
    :type end: datetime
    :param period : time-window
    :type period: int
    :return: relevant four dataframes
    :rtype: pd.dataframe
    """
    #read in data -- adj close price
    df = get_data(syms, pd.date_range(start-dt.timedelta(days = period*3), end))
    #normalize price
    df = df/df.iloc[0,:]
    #remove spy
    df = df.iloc[:,1:]
    #sma compute
    sma = df.rolling(period, min_periods = period).mean()
    #rolling std compute
    rolling_std = df.rolling(period, min_periods = period).std()
    #upper bollinger band
    top_band = sma + 2*rolling_std
    #lower bollinger ban
    bottom_band = sma - 2*rolling_std
    #bb %b compute
    bbp = (df-bottom_band)/(top_band-bottom_band)
    bbp = bbp.loc[start:,:]

    return sma, top_band, bottom_band, bbp

#--------------------------------------- MACD - Trends --------------------------------------------
# def ema (syms, start, end, period):
#     """This function calculates ema indicator so that we could easily calculate macd below      

#     :param syms: symbols
#     :type syms: list
#     :param start: start date
#     :type start: datetime
#     :param end: end date
#     :type end: datetime
#     :param period: period for ema
#     :type period: int
#     :return: specified period ema dataframe
#     :rtype: pd.dataframe
#     """
#     #read in data -- adj close price
#     df = get_data(syms, pd.date_range(start, end))
#     #normalize price
#     # df = df/df.iloc[0,:]
#     #remove spy
#     df = df.iloc[:,1:]
#     #sam compute
#     sma = df.rolling(period, min_periods = period).mean()
#     #multipler compute
#     multiplier = 2/(period+1)
#     #inital ema df
#     ema = df.copy()
#     ema.loc[:, :] = np.nan

#     #calculate ema
#     ema.iloc[period:, :] = (df.iloc[period:, :] - sma.iloc[period-1, :] ) * multiplier +  sma.iloc[period-1, :]
#     #handle first ema (equal to sma)
#     # ema.iloc[:period-1,:] = np.nan
#     ema.iloc[period-1,:] = sma.iloc[period-1,:]

#     return ema

def macd(syms, start, end):
    """This function calculates macd indicator (12-day and 26-day EMA period)

    :param syms: symbols
    :type syms: list
    :param start: start date
    :type start: datetime
    :param end: end date
    :type end: datetime
    :return: relevant four dataframes (12-day, 26-day, 9-day(singal) and MACD histogram df)
    :rtype: pd.dataframe
    """

    #read in data -- adj close price
    df = get_data(syms, pd.date_range(start-dt.timedelta(days = 26*2), end))
    df = df.iloc[:,1:] #remove spy
    #normalize price
    df = df/df.iloc[0,:]
    #12-day ema
    ema_12 = df.ewm(span = 12, min_periods = 12, adjust = False).mean()
    #26-day ema
    ema_26 = df.ewm(span = 26, min_periods = 26, adjust = False).mean()
    #macd line
    macd = ema_12 - ema_26
    #9-day ema (signal line)
    signal = macd.ewm(span = 9, min_periods = 9, adjust = False).mean()
    
    #macd hisogram
    macd_hist =  macd-signal
    macd_hist = macd_hist.loc[start:,:]
    macd = macd.loc[start:, :]
    return ema_12, ema_26, macd, signal, macd_hist
    # return macd 
#--------------------------------------- OBV - Volume --------------------------------------------
def obv(syms, start, end):
    """This function calculates obv indicator

    :param syms: symbols
    :type syms: list
    :param start: start date
    :type start: datetime
    :param end: end date
    :return: obv dataframe
    :rtype: pd.dataframe
    """
    #read in data -- adj close price
    price = get_data(syms, pd.date_range(start, end))
    #normalize price
    price = price/price.iloc[0,:]
    #remove spy
    price = price.iloc[:,1:]
    #read in data -- volumne
    vol = get_data(syms, pd.date_range(start, end), colname = 'Volume')
    #remove spy
    vol = vol.iloc[:,1:]
    #normalize volume
    # vol = vol/vol.iloc[0,:]
    #initialize obv df
    obv = price.copy()
    obv.loc[:,:] = 0
    #compute obv
    obv = vol.where(price > price.shift(1), -vol.where(price < price.shift(1), 0)).cumsum()
    obv = obv.loc[start:,:]
    
    return obv, vol
#--------------------------------------- CCI - Others --------------------------------------------
def cci(syms, start, end, period=20):
    """This function calculates cci indicator

    :param syms: symbols
    :type syms: list
    :param start: start date
    :type start: datetime
    :param end: end date
    :param period: time-window
    :type period: int
    :return: relevant dataframes
    :rtype: pd.dataframe
    """
    #read in data -- adj close price, high price and low price
    adjclose = get_data(syms, pd.date_range(start-dt.timedelta(days = period*3), end))
    adjclose = adjclose.iloc[:,1:]
    adjclose = adjclose/adjclose.iloc[0,:] #normalize
    close = get_data(syms, pd.date_range(start, end), colname = 'Close')
    close = close.iloc[:,1:] #remove spy
    close = close/close.iloc[0,:] #normalize
    #compute adjust ratio
    adjratio = adjclose/close
    high = get_data(syms, pd.date_range(start, end), colname = 'High')
    high = high.iloc[:,1:] #remove spy
    high = high/high.iloc[0,:] #normalize
    adjhigh = high * adjratio #adjust high
    low = get_data(syms, pd.date_range(start, end), colname = 'Low')
    low = low.iloc[:,1:] #remove spy
    low = low/low.iloc[0,:] #normalize
    adjlow = low * adjratio #adjust low
    #compute typical price
    tp = (adjclose + adjhigh + adjlow)/3
    #sma of tp
    sma_tp = tp.rolling(period, min_periods = period).mean()
    #compute mean deviation
    mad = lambda x: np.fabs(x-x.mean()).mean()
    md = tp.rolling(period, min_periods = period).apply(mad, raw = True)
    # tp_std = tp.rolling(period, min_periods = period).std()
    #compute cci
    cci_val = (tp - sma_tp)/(0.015 * md)
    cci_val = cci_val.loc[start:, :]

    return cci_val, tp, sma_tp

def author():
    return 'yjiang347'

#-------------------------Define main function----------------------
def main(syms, start_date, end_date):
    """This function plot those 5 indicators

    :param syms: symbols
    :type syms: list
    :param start_date: start date
    :type start: datetime
    :param end_date: end date
    """
    #set general parameter for all plots
    start_date = '2008-01-01'
    end_date = '2009-12-31'
    syms = ['JPM']
    start_date = start_date-dt.timedelta(days = 60)
    adjclose = get_data(syms, pd.date_range(start_date, end_date))
    adjclose = adjclose.iloc[:,1:] #remove spy
    adjclose = adjclose/adjclose.iloc[0,:] #normalize

#===============================================CCI Plotting================================================
    #get data
    cci_val, tp, sma_tp = cci(syms, start_date, end_date, 20)
    #----------------the whole plot setting----------------------
    fig, axs = plt.subplots(2, sharex = True, gridspec_kw={'height_ratios': [2, 1.3]}, figsize = (16,8))
    fig.suptitle('Commodity Channel Index (CCI) in Period of 20 Days of JPM')
    fig.subplots_adjust(hspace = 0.15)
    plt.setp(axs[1].get_xticklabels(),rotation = 30, ha = 'right')
    
    #---------------------subplot1: price-----------------------
    axs[0].plot(adjclose, label = "Adjusted Close")
    axs[0].plot(tp, label = 'Typical Price')
    axs[0].plot(sma_tp, label = 'SMA')
    # axs[0].set_ylim((0,2))
    axs[0].grid(color='gray', ls = '-.', lw = 0.25)
    axs[0].set_title('Price vs. Date')
    axs[0].set_ylabel('Normalized Price',labelpad = 20)
    axs[0].legend(loc = 'lower left')
    axs[0].tick_params(axis = 'x', bottom = False)
    #----------------------subplot2: cci-------------------------
    axs[1].plot(cci_val)
    axs[1].set_title('CCI vs. Date')
    axs[1].grid(color='gray', ls = '-.', lw = 0.25)
    axs[1].plot(cci_val.index, [100]*cci_val.shape[0], color = 'darkred', label = '100-Line', lw= 1)
    axs[1].plot(cci_val.index, [-100]*cci_val.shape[0], color = 'red', label = '-100-Line', lw = 1)
    axs[1].set_ylabel('CCI Value', labelpad = 5)

    axs[1].set_xlabel('Date', fontsize = 12 , labelpad = 5)
    axs[1].set_ylim((-300,350))
    axs[1].set_xlim(('2008-01-01', '2010-01-01'))
   
    #--------------------------add text to subplot2-------------------
    #box ref: https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/fancytextbox_demo.html#sphx-glr-gallery-text-labels-and-annotations-fancytextbox-demo-py
    axs[1].text('2008-04-10',-170, "CCI = -100", size = 10, rotation = 0, ha='center', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8) ))
    axs[1].text('2009-06-10',145, "CCI = 100", size = 10, rotation = 0, ha='center', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8) ))
    
    #-------------save figure in current file--------------------------
    plt.savefig('CCI_Plot')
    #===========================================CCI Plot End========================================================

    #==================================================OBV Plotting=================================================
    #get data
    obv_val, vol = obv(syms, start_date, end_date)

    #-------------------the whole plot---------------------------
    fig1, axs1 = plt.subplots(2, sharex = True, gridspec_kw={'height_ratios': [2, 1.3]}, figsize = (16,8))
    fig1.suptitle('On Balance Volume (OBV) of JPM')
    fig1.subplots_adjust(hspace = 0.15)
    plt.setp(axs1[1].get_xticklabels(),rotation = 30, ha = 'right')

    #-------------------subplot1: price---------------------------
    axs1[0].plot(adjclose)
    axs1[0].set_xlim(('2008-01-01', '2010-01-01'))
    # axs1[0].set_ylim((0,2))
    axs1[0].grid(color='gray', ls = '-.', lw = 0.25)
    axs1[0].set_title('Normalized Adjusted Close Price vs. Date')
    axs1[0].set_ylabel('Normalized Adjusted Close Price',labelpad = 29)
    axs1[0].tick_params(axis = 'x', bottom = False)

    # #---------------------subplot2: volume------------------------
    # x = np.arange(vol.shape[0])
    # axs1[1].bar(vol.index, vol.iloc[:,0].values/100000000)
    # axs1[1].set_title('Volume vs. Date')
    # axs1[1].grid(color='gray', ls = '-.', lw = 0.25)
    # axs1[1].set_ylabel('Volume (1e8)',labelpad = 20)
    # axs1[1].tick_params(axis = 'x', bottom = False)

    #---------------------subplot2: OBV---------------------------
    axs1[1].plot(obv_val/1000000, color = "red")
    axs1[1].set_title('OBV vs. Date')
    axs1[1].grid(color='gray', ls = '-.', lw = 0.25)
    axs1[1].set_ylabel('OBV (in millions)',labelpad = 10)
    axs1[1].set_xlabel('Date', fontsize = 12 , labelpad = 5)
    

    # -------------save figure in current file--------------------------
    plt.savefig('OBV_Plot')
    #===========================================OBV Plot End========================================================
    
    #==============================================MACD Plot========================================================
    #get data
    ema_12, ema_26, macd_val, signal, macd_hist = macd(syms, start_date, end_date)

    #----------------the whole plot setting----------------------
    fig2, axs2 = plt.subplots(2, sharex = True, gridspec_kw={'height_ratios': [2, 1.4]}, figsize = (16,8))
    fig2.suptitle('Moving Average Convergence/Divergence (MACD) for JPM')
    fig2.subplots_adjust(hspace = 0.15)
    plt.setp(axs2[1].get_xticklabels(),rotation = 30, ha = 'right')
    
    #---------------------subplot1: price-----------------------
    axs2[0].plot(adjclose, label = "Adjusted Close")
    axs2[0].plot(ema_12, label = '12-day EMA')
    axs2[0].plot(ema_26, label = '26-day EMA')
    # axs2[0].set_ylim((0,2))
    axs2[0].grid(color='gray', ls = '-.', lw = 0.25)
    axs2[0].set_title('Normalized Price vs. Date')
    axs2[0].set_ylabel('Normalized Price',labelpad = 20)
    axs2[0].legend(loc = 'lower left')
    axs2[0].tick_params(axis = 'x', bottom = False)
    #----------------------subplot2: macd-------------------------
    axs2[1].plot(macd_val, label = 'MACD')
    axs2[1].plot(signal, label = 'Signal Line')
    axs2[1].bar(macd_hist.index, macd_hist.iloc[:,0].values, color = 'grey', label = 'MACD-Histogram')
    axs2[1].set_title('MACD vs. Date')
    axs2[1].grid(color='gray', ls = '-.', lw = 0.25)
    axs2[1].set_ylabel('MACD', labelpad = 15)
    axs2[1].set_xlabel('Date', fontsize = 12 , labelpad = 5)
    axs2[1].set_xlim(('2008-01-01', '2010-01-01'))
    axs2[1].legend(loc = 'lower left')

    #-------------save figure in current file--------------------------
    plt.savefig('MACD_Plot')
#===============================================MACD Plot end ================================================

#===============================================BBP Plotting================================================
# get data
    sma, top_band, bottom_band, bbp_val = bbp(syms, start_date, end_date, 20)
    #----------------the whole plot setting----------------------
    fig3, axs3 = plt.subplots(2, sharex = True, gridspec_kw={'height_ratios': [2, 1.3]}, figsize = (16,8))
    fig3.suptitle('Bollinger Bands %b in Period of 20 Days of JPM')
    fig3.subplots_adjust(hspace = 0.15)
    plt.setp(axs3[1].get_xticklabels(),rotation = 30, ha = 'right')
    
    #---------------------subplot1: price-----------------------
    axs3[0].plot(adjclose, label = "Adjusted Close")
    axs3[0].plot(sma, label = 'SMA')
    axs3[0].plot(bottom_band, label = 'Bottom BB')
    axs3[0].plot(top_band, label = 'Upper BB')
    # axs3[0].set_ylim((10,50))
    axs3[0].grid(color='gray', ls = '-.', lw = 0.25)
    axs3[0].set_title('Normalized Price vs. Date')
    axs3[0].set_ylabel('Normalized Price',labelpad = 20)
    axs3[0].legend(loc = 'lower left')
    axs3[0].tick_params(axis = 'x', bottom = False)
    #----------------------subplot2: bbp-------------------------
    axs3[1].plot(bbp_val)
    axs3[1].set_title('BBP vs. Date')
    axs3[1].grid(color='gray', ls = '-.', lw = 0.25)
    axs3[1].plot(bbp_val.index, [1]*bbp_val.shape[0], color = 'darkred', label = '1-Line', lw= 1)
    axs3[1].plot(bbp_val.index, [0]*bbp_val.shape[0], color = 'red', label = '-1-Line', lw = 1)
    axs3[1].set_ylabel('BBP Value', labelpad = 5)
    axs3[1].set_xlabel('Date', fontsize = 12 , labelpad = 5)
    axs3[1].set_xlim(('2008-01-01', '2010-01-01'))
   
    #--------------------------add text to subplot2-------------------
    #box ref: https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/fancytextbox_demo.html#sphx-glr-gallery-text-labels-and-annotations-fancytextbox-demo-py
    axs3[1].text('2008-04-10',-0.15, "BBP = 0.00", size = 10, rotation = 0, ha='center', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8) ))
    axs3[1].text('2009-06-10',1.1, "BBP = 1.00", size = 10, rotation = 0, ha='center', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8) ))
    
    #-------------save figure in current file--------------------------
    plt.savefig('BBP_Plot')
   
 #==================================================RSI Plotting=================================================
    #get data
    rsi_val = rsi(syms, start_date, end_date,14)

    #-------------------the whole plot---------------------------
    fig4, axs4 = plt.subplots(2, sharex = True, gridspec_kw={'height_ratios': [2, 1.3]}, figsize = (16,8))
    fig4.suptitle('Relative Strength Index (RSI) in 14-day Period of JPM')
    fig4.subplots_adjust(hspace = 0.15)
    plt.setp(axs4[1].get_xticklabels(),rotation = 30, ha = 'right')

    #-------------------subplot1: price---------------------------
    axs4[0].plot(adjclose)
    axs4[0].set_xlim(('2008-01-01', '2010-01-01'))
    # axs4[0].set_ylim((10,50))
    axs4[0].grid(color='gray', ls = '-.', lw = 0.25)
    axs4[0].set_title('Normalized Adjusted Close Price vs. Date')
    axs4[0].set_ylabel('Normalized Adjusted Close Price',labelpad = 15)
    axs4[0].tick_params(axis = 'x', bottom = False)

    #---------------------subplot2: rsi---------------------------
    axs4[1].plot(rsi_val,color = 'forestgreen')
    axs4[1].set_title('RSI vs. Date')
    axs4[1].grid(color='gray', ls = '-.', lw = 0.25)
    axs4[1].set_ylabel('RSI',labelpad = 20)
    axs4[1].set_xlabel('Date', fontsize = 12 , labelpad = 5)
    axs4[1].plot(rsi_val.index, [70]*rsi_val.shape[0], color = 'darkred', label = '70-Threshold', lw= 1)
    axs4[1].plot(rsi_val.index, [30]*rsi_val.shape[0], color = 'red', label = '30-Threshold', lw = 1)
    axs4[1].plot(rsi_val.index, [50]*rsi_val.shape[0], color = 'grey',ls = '--', label = 'CenterLine', lw = 1)
    axs4[1].legend(loc = 'upper left', ncol = 3)
    #--------------------------add text to subplot2-------------------
    #box ref: https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/fancytextbox_demo.html#sphx-glr-gallery-text-labels-and-annotations-fancytextbox-demo-py
    axs4[1].text('2008-09-01',21, "Oversold(30)", size = 10, rotation = 0, ha='center', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8) ))
    axs4[1].text('2008-12-01',76, "Overbought(70)", size = 10, rotation = 0, ha='center', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8) ))
    

    # -------------save figure in current file--------------------------
    plt.savefig('RSI_Plot')

if __name__ == "__main__":
    
    main()