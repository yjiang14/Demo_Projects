""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Yan Jiang (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: yjiang347 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 903250461 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import random  
import QLearner as ql
import indicators as ind
import marketsimcode as mk
import matplotlib.pyplot as plt
import ManualStrategy as ms
     		  		  		    	 		 		   		 		  
import pandas as pd 
import numpy as np 		  	   		     		  		  		    	 		 		   		 		  
import util as ut  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # constructor  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		     		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		     		  		  		    	 		 		   		 		  
        self.commission = commission 

        self.bbp_bin = 10
        self.rsi_bin = 8
        self.cci_bin = 8
        self.learner = ql.QLearner(num_states = self.bbp_bin*self.rsi_bin*self.cci_bin, num_actions = 3, alpha= 0.21, gamma = 0.9, rar = 0.65, radr = 0.9, dyna = 0, verbose = self.verbose)	
    
        		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        symbol="JPM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),  		  	   		     		  		  		    	 		 		   		 		  
        sv=100000,  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 12/31/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        # add your code to do learning here  
        

        #--------------------------read in dataset-------------------------------------------
        dates = pd.date_range(sd, ed)
        price = ut.get_data([symbol], dates)
        price = price.drop(columns = ['SPY'], axis = 1)

        #-------------------------Create indicators df---------------------------------------
        """ Selected 3 indicators: bbp, macd, rsi"""
        #bbp compute
        sma, top_band, bottom_band, bbp = ind.bbp([symbol], sd, ed, period=20)
        bbp_norm = (bbp-bbp.mean(axis = 0))/bbp.std(axis=0)
       
        #macd compute
        # ema_12, ema_26, macd, signal, macd_hist = ind.macd([symbol], sd, ed)

        """
           macd_hist >= 0 --> 1 otherwise --> -1
           Multiply macd
        """
        # # macd_norm = (macd-macd.mean(axis = 0))/macd.std(axis=0)
        # macd_df = pd.DataFrame()
        # macd_df['diff'] = macd_hist.iloc[:,0]
        # macd_df['cross'] = [1 if x >= 0 else -1 for x in macd_hist.values]
        # # macd_df['cross_diff'] = macd_df['cross'].diff()
        # # macd_df.iloc[0,-1] = 0 #change first nan to 0 after .diff() 
        # macd_df['signal'] = macd_df.iloc[:,0] * macd_df.iloc[:,1]

        # macd_df['macd_norm'] = (macd_df['signal']-macd_df['signal'].mean(axis = 0))/macd_df['signal'].std(axis = 0)
       
        #rsi compute
        rsi = ind.rsi([symbol], sd, ed, period= 14)
        rsi_norm = (rsi - rsi.mean(axis=0))/rsi.std(axis = 0)

        #cci compute
        cci_val, tp, sma_tp = ind.cci([symbol], sd, ed, 20)
        cci_norm = (cci_val-cci_val.mean(axis= 0))/cci_val.std(axis= 0)


        #----------------------Discrete each indicator to 9 status---------------------
       
        ind_df = price.copy()
        
        bbp_out, bbp_bins = pd.qcut(bbp_norm.iloc[:, 0], self.bbp_bin, labels = False, retbins = True)
        ind_df["bbp"] = list(bbp_out)
        self.bbp_bins = list(bbp_bins)

        rsi_out, rsi_bins = pd.qcut(rsi_norm.iloc[:, 0], self.rsi_bin, labels = False, retbins = True)
        ind_df["rsi"] = list(rsi_out)
        self.rsi_bins = list(rsi_bins)

        cci_out, cci_bins = pd.qcut(cci_norm.iloc[:, 0], self.cci_bin, labels = False, retbins = True)
        ind_df["cci"] = list(cci_out)
        self.cci_bins = list(cci_bins)

        ind_df["sum"] = ind_df['rsi']*100 + ind_df['bbp']*10 + ind_df['cci'] 

        # #--------------------------------------discretiza ind_df-------------------------------------       
        state = pd.DataFrame(0, index = price.index, columns = price.columns)
        count = 0
        for i in range(self.bbp_bin):
            for j in range(self.rsi_bin):
                for k in range(self.cci_bin):
                    state[ind_df["sum"] == (i*100 + j*10 + k)] = count        
                    count += 1
        ind_df["state"] = state.iloc[:].values
  	   		     		  		  		    	 		 		   		 		  
        if self.verbose:
            print (ind_df)
        #-----------------------------------Q-learner Process ------------------------------------------
        """
            actions: long --0, short--1,  --3
            state: 3*3*3 = 27
        """
          
        portvals = price.copy()
        portvals.iloc[:] = 0 #initialize portfolio value
        df_trades = price.copy() #create trading df
        df_trades.iloc[:,:] = 0 #initialize trading df
        holding = price.copy() #create holding df
        holding.iloc[:,0] = float('nan') #intialize holding df
       
        count = 0
        total_days = price.shape[0]
       
       
        cum_ret = -20000
        converge = False

        while (not converge) & (count <= 10):

            #the first day
            state = ind_df.iloc[0,-1] # get first state
            action = self.learner.querysetstate(state) # get action based on first state

            if action == 0:
                holding.iloc[0] = 1000
            elif action == 1:
                holding.iloc[0] = -1000
            else:
                holding.iloc[0] = 0

            df_trades.iloc[0] = holding.iloc[0]

            portvals.iloc[0] = 1

            for count_day in range(1, total_days):
                # reward compute
                if df_trades.iloc[count_day-1].values != 0:
                    fees = self.commission + abs(df_trades.iloc[count_day-1].values) * self.impact * price.iloc[count_day-1]
                else:
                    fees = pd.Series(0)
                reward = ((price.iloc[count_day] - price.iloc[count_day-1])*holding.iloc[count_day-1]-fees.iloc[0])/sv

                # print(reward)
                state = ind_df.iloc[count_day, -1]

                #update Q table and get new action
                action = self.learner.query(state, reward)

                if action == 0:
                    holding.iloc[count_day] = 1000
                elif action == 1:
                    holding.iloc[count_day] = -1000
                else:
                    holding.iloc[count_day] = holding.iloc[count_day-1]
                
                #update df_trades  
                df_trades.iloc[count_day] = holding.iloc[count_day] - holding.iloc[count_day-1]
                #update portvals df
                portvals.iloc[count_day] = portvals.iloc[count_day-1] + reward
                

                # if df_trades.iloc[count_day, 0] != 0:
                #     fees = self.commission + abs(df_trades.iloc[count_day, 0]) * self.impact * price.iloc[count_day]
                # else:
                #     fees = 0

                # df_trades.iloc[count_day, 1] = (0-(df_trades.iloc[count_day,0] * price.iloc[count_day]) - fees).iloc[0]

                # holding.iloc[count_day,1] = holding.iloc[count_day-1,1] + df_trades.iloc[count_day,1]

                # portvals.iloc[count_day] = (holding.iloc[count_day, :] * new_price.iloc[count_day, :]).sum(axis=0)

                # reward = (portvals.iloc[count_day] - portvals.iloc[count_day-1])/portvals.iloc[0]

            #compute  this loop cum return
            curr_cum_ret = (portvals.iloc[-1]/portvals.iloc[0]) -1
            curr_cum_ret = curr_cum_ret[0]
            

            if self.verbose:
                print(curr_cum_ret)

            count += 1

            #check converge
            if curr_cum_ret <= cum_ret and curr_cum_ret >= 0 and count >= 3:
                converge = True
            else:
                cum_ret=curr_cum_ret
        
        if self.verbose:
          print (f'this is holding{holding}')
          print(f'this is trading{df_trades}')
          print(portvals)


        # example usage of the old backward compatible util function  		  	   		     		  		  		    	 		 		   		 		  
        # syms = [symbol]  		  	   		     		  		  		    	 		 		   		 		  
        # dates = pd.date_range(sd, ed)  		  	   		     		  		  		    	 		 		   		 		  
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
        # prices = prices_all[syms]  # only portfolio symbols  		  	   		     		  		  		    	 		 		   		 		  
        # prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		     		  		  		    	 		 		   		 		  
        	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        # example use with new colname  		  	   		     		  		  		    	 		 		   		 		  
        # volume_all = ut.get_data(  		  	   		     		  		  		    	 		 		   		 		  
        #     syms, dates, colname="Volume"  		  	   		     		  		  		    	 		 		   		 		  
        # )  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
        # volume = volume_all[syms]  # only portfolio symbols  		  	   		     		  		  		    	 		 		   		 		  
        # volume_SPY = volume_all["SPY"]  # only SPY, for comparison later  		  	   		     		  		  		    	 		 		   		 		  
        # if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
        #     print(volume)  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		     		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        symbol="JPM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),  		  	   		     		  		  		    	 		 		   		 		  
        sv=100000,  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 12/31/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		     		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		     		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		     		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		     		  		  		    	 		 		   		 		  
        """  
        #--------------------------read in dataset-------------------------------------------
        dates = pd.date_range(sd, ed)
        price = ut.get_data([symbol], dates)
        price = price.drop(columns = ['SPY'], axis = 1)

        #-------------------------Create indicators df---------------------------------------
        """ Selected 3 indicators: bbp, macd, rsi"""
        #bbp compute
        sma, top_band, bottom_band, bbp = ind.bbp([symbol], sd, ed, period=20)
        bbp_norm = (bbp-bbp.mean(axis = 0))/bbp.std(axis=0)
       
        #macd compute
        # ema_12, ema_26, macd, signal, macd_hist = ind.macd([symbol], sd, ed)

        #rsi compute
        rsi = ind.rsi([symbol], sd, ed, period= 14)
        rsi_norm = (rsi - rsi.mean(axis=0))/rsi.std(axis = 0)

        #cci compute
        cci_val, tp, sma_tp = ind.cci([symbol], sd, ed, 20)
        cci_norm = (cci_val-cci_val.mean(axis= 0))/cci_val.std(axis= 0)

        #--------------------------------------discretiza ind_df based training bins-------------------------------------
        state_bbp = pd.DataFrame(0, index = price.index,columns = price.columns)
        count = 0
        for i in range(len(self.bbp_bins)-1):
            state_bbp[(bbp_norm >= self.bbp_bins[i]) & ( bbp_norm <= self.bbp_bins[i+1])] = count
            count += 1

        state_rsi = pd.DataFrame(0, index = price.index,columns = price.columns)
        count = 0
        for i in range(len(self.rsi_bins)-1):
            state_rsi[(rsi_norm >= self.rsi_bins[i]) & ( rsi_norm <= self.rsi_bins[i+1])] = count
            count += 1

        state_cci = pd.DataFrame(0, index = price.index,columns = price.columns)
        count = 0
        for i in range(len(self.cci_bins)-1):
            state_cci[(cci_norm >= self.cci_bins[i]) & ( cci_norm <= self.cci_bins[i+1])] = count
            count += 1
            
        ind_df = pd.DataFrame(index = price.index)
        ind_df["bbp"] = state_bbp.iloc[:].values
        ind_df["rsi"] = state_rsi.iloc[:].values
        ind_df["cci"] = state_cci.iloc[:].values
        ind_df["sum"] = ind_df['rsi']*100 + ind_df['bbp']*10 + ind_df['cci'] 

        state = pd.DataFrame(0, index = price.index, columns = price.columns)
        count = 0
        for i in range(self.bbp_bin):
            for j in range(self.rsi_bin):
                for k in range(self.cci_bin):
                    state[ind_df["sum"] == (i*100 + j*10 + k)] = count        
                    count += 1
        ind_df["state"] = state.iloc[:].values
        
        df_trades = pd.DataFrame(0, index = price.index, columns = price.columns) #create and initialize trading df
        holding = pd.DataFrame(0, index = price.index, columns = price.columns) #create and initilize holding df
      

        count = 0
        total_days = price.shape[0]
        

        # Qlearner implementation
        for count_day in range(0, total_days):
            # get state
            state = ind_df.iloc[count_day, -1]
    
            #get new action
            action = self.learner.querysetstate(state)

            if action == 0:
                holding.iloc[count_day] = 1000
            elif action == 1:
                holding.iloc[count_day] = -1000
            else:
                holding.iloc[count_day] =holding.iloc[count_day-1]

            if count_day > 0:        
                df_trades.iloc[count_day] = holding.iloc[count_day] - holding.iloc[count_day-1]
            else:
                df_trades.iloc[count_day] = holding.iloc[count_day]
            
            count_day += 1 

           
        # print(df_trades)   
        return df_trades

        # here we build a fake set of trades  		  	   		     		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		  	   		     		  		  		    	 		 		   		 		  
        # dates = pd.date_range(sd, ed)  		  	   		     		  		  		    	 		 		   		 		  
        # prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
        # trades = prices_all[[symbol,]]  # only portfolio symbols  		  	   		     		  		  		    	 		 		   		 		  
        # trades_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		     		  		  		    	 		 		   		 		  
        # trades.values[:, :] = 0  # set them all to nothing  		  	   		     		  		  		    	 		 		   		 		  
        # trades.values[0, :] = 1000  # add a BUY at the start  		  	   		     		  		  		    	 		 		   		 		  
        # trades.values[40, :] = -1000  # add a SELL  		  	   		     		  		  		    	 		 		   		 		  
        # trades.values[41, :] = 1000  # add a BUY  		  	   		     		  		  		    	 		 		   		 		  
        # trades.values[60, :] = -2000  # go short from long  		  	   		     		  		  		    	 		 		   		 		  
        # trades.values[61, :] = 2000  # go long from short  		  	   		     		  		  		    	 		 		   		 		  
        # trades.values[-1, :] = -1000  # exit on the last day  		  	   		     		  		  		    	 		 		   		 		  
        # if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
        #     print(type(trades))  # it better be a DataFrame!  		  	   		     		  		  		    	 		 		   		 		  
        # if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
        #     print(trades)  		  	   		     		  		  		    	 		 		   		 		  
        # if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
        #     print(prices_all)  		  	   		     		  		  		    	 		 		   		 		  
        # return trades  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    #implement author func
    def author(self):
        return 'yjiang347'
 	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    # print("One does not simply think up a strategy")  		  	   		     		  		  		    	 		 		   		 		  
    learner = StrategyLearner(verbose=False, impact=0.01, commission=0)
    learner.add_evidence( symbol="JPM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),  		  	   		     		  		  		    	 		 		   		 		  
        sv=100000, )
    df_trades = learner.testPolicy(symbol="JPM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),  		  	   		     		  		  		    	 		 		   		 		  
        sv=100000)

    portvals = mk.compute_portvals(orders_file = df_trades, start_val=100000, commission=0, impact=0.01)
    # portvals2 = mk.compute_portvals(orders_file = df_trades1.to_frame(), start_val=100000, commission=9.95, impact=0.005)
    # print(portvals_1, portvals2)
    portvals = (portvals/portvals.iloc[0])

    ## II. create bench trades df
    price = ut.get_data(["JPM"] ,pd.date_range(dt.datetime(2008,1,1),dt.datetime(2009, 12, 31)))
    price = price.iloc[:,1:]#remove spy
    bench_trades = price.copy()
    bench_trades.iloc[0,:] = 1000
    bench_trades.iloc[1:,:] = 0
    ## III. get bench daily portfolio value 
    bench_portvals = mk.compute_portvals(orders_file = bench_trades, start_val=100000, commission=0, impact=0.005)
    bench_portvals = bench_portvals/bench_portvals.iloc[0,:]
    ###IV. get manual strategy
    ms_df_trades = ms.testPolicy(symbol = "JPM", sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), sv = 100000)

    ms_portvals = mk.compute_portvals(orders_file = ms_df_trades, start_val=100000, commission=9.95, impact=0.005)
    ms_portvals = ms_portvals/ms_portvals.iloc[0]	 

    # fig = plt.figure(figsize = (12,8))
    # plt.plot(portvals, label = "StrategyLearner", color = 'red')
    # plt.plot(bench_portvals, label = "Benchmark", color = 'green')
    # plt.plot(ms_portvals, label = 'ManualStrategy', color = 'Black')
    # plt.title('Benchmark vs. Strategy Learner vs. Manual Strategy \n(Portfolio Value)')
    # plt.ylabel('Normalized Portfolio Value')
    # plt.xlabel('Date')
    # plt.legend(loc = 'upper left')
    # plt.xticks(rotation = 30)
    # # plt.savefig('portval.png')
   