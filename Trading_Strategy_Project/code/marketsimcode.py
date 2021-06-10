""""""  		  	   		     		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
import os  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def compute_portvals(  		  	   		     		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		     		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		     		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		     		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object or dataframe		  	   		     		  		  		    	 		 		   		 		  
    :type orders_file: str or file object or dataframe  		  	   		     		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		     		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		     		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		  	   		     		  		  		    	 		 		   		 		  
    # TODO: Your code here

   
    # (1) Extract symbols and dates for get_date() function from order_file
    if isinstance(orders_file, str):
        orders = pd.read_csv(
                orders_file,
                index_col = "Date",
                parse_dates = True
        )

        ### I. get start and end date for orders
        start_date = min(orders.index)
        end_date = max(orders.index)

        ### II. get symbols in orders
        symbols = list(orders['Symbol'].drop_duplicates())

        #(2) get prices dataframe using  I and II info
        prices = get_data(symbols, pd.date_range(start_date, end_date)) 
        prices['Cash'] = 1.0
        prices = prices.iloc[:,1:]

        #(3) get tradings dataframe
        tradings = prices.copy()
        tradings.loc[:, :] = 0 #initialize all cell to be 0

        ### I. Update stock trading shares
        fees = tradings.iloc[:,0].copy()
        for i in range(orders.shape[0]):
            for j in range(tradings.shape[0]):
                if orders.index[i] == tradings.index[j]:
                    if orders.iloc[i,1] == 'BUY':
                        tradings.loc[orders.index[i],orders.iloc[i,0]] += +int(orders.iloc[i,2]) #cum is combination of same date with same stocking trading
                        fees.loc[orders.index[i]] +=  commission + prices.loc[orders.index[i], orders.iloc[i,0]]*impact*int(orders.iloc[i,2])
                    elif orders.iloc[i,1] == 'SELL':
                        tradings.loc[orders.index[i],orders.iloc[i,0]] += -int(orders.iloc[i,2])
                        fees.loc[orders.index[i]] +=  commission + prices.loc[orders.index[i], orders.iloc[i,0]]*impact*int(orders.iloc[i,2])
        ### II. Update Cash column
        tradings['Cash'] = 0-(prices * tradings).sum(axis = 1) - fees

        # (3) get holdings dataframe
        holdings = tradings.copy()
        holdings.loc[:,:] = 0 #initialize all cell to be 0
        holding = holdings.iloc[0,:].copy() #initialize one row in holding

        ### I Update holdings shares
        for date in tradings.index:
            holding += tradings.loc[date,:]
            holdings.loc[date, :] = holding

        ### II. Update holdings cash
        holdings['Cash'] = start_val + holdings['Cash']

        ### III. acccount commissions and market impact for cash col

        # (4) get values dataframe
        values = tradings.copy()
        values.iloc[:,:] = 0 # initialize each cell to be 0
    
        values = prices * holdings

        # (5)get portvals with dataframe type
        portvals = pd.DataFrame(values.iloc[:,:-1].sum(axis = 1) + values['Cash'])

        return portvals

    elif hasattr(orders_file, 'read'):
        orders_path = orders_file.name
        orders = pd.read_csv(
                orders_path,
                index_col = "Date",
                parse_dates = True
        )

        ### I. get start and end date for orders
        start_date = min(orders.index)
        end_date = max(orders.index)

        ### II. get symbols in orders
        symbols = list(orders['Symbol'].drop_duplicates())
        
        #(2) get prices dataframe using  I and II info
        prices = get_data(symbols, pd.date_range(start_date, end_date)) 
        prices['Cash'] = 1.0
        prices = prices.iloc[:,1:]

        #(3) get tradings dataframe
        tradings = prices.copy()
        tradings.loc[:, :] = 0 #initialize all cell to be 0

        ### I. Update stock trading shares
        fees = tradings.iloc[:,0].copy()
        for i in range(orders.shape[0]):
            for j in range(tradings.shape[0]):
                if orders.index[i] == tradings.index[j]:
                    if orders.iloc[i,1] == 'BUY':
                        tradings.loc[orders.index[i],orders.iloc[i,0]] += +int(orders.iloc[i,2]) #cum is combination of same date with same stocking trading
                        fees.loc[orders.index[i]] +=  commission + prices.loc[orders.index[i], orders.iloc[i,0]]*impact*int(orders.iloc[i,2])
                    elif orders.iloc[i,1] == 'SELL':
                        tradings.loc[orders.index[i],orders.iloc[i,0]] += -int(orders.iloc[i,2])
                        fees.loc[orders.index[i]] +=  commission + prices.loc[orders.index[i], orders.iloc[i,0]]*impact*int(orders.iloc[i,2])
        ### II. Update Cash column
        tradings['Cash'] = 0-(prices * tradings).sum(axis = 1) - fees

         # (3) get holdings dataframe
        holdings = tradings.copy()
        holdings.loc[:,:] = 0 #initialize all cell to be 0
        holding = holdings.iloc[0,:].copy() #initialize one row in holding

        ### I Update holdings shares
        for date in tradings.index:
            holding += tradings.loc[date,:]
            holdings.loc[date, :] = holding

        ### II. Update holdings cash
        holdings['Cash'] = start_val + holdings['Cash']

        ### III. acccount commissions and market impact for cash col

        # (4) get values dataframe
        values = tradings.copy()
        values.iloc[:,:] = 0 # initialize each cell to be 0
    
        values = prices * holdings

        # (5)get portvals with dataframe type
        portvals = pd.DataFrame(values.iloc[:,:-1].sum(axis = 1) + values['Cash'])

        return portvals


    elif isinstance(orders_file, pd.DataFrame):
        tradings = orders_file

        start_date = min(tradings.index)
        end_date = max(tradings.index)

        #get symbols in df
        symbols = [col for col in tradings.columns]
    
        #get prices
        prices = get_data(symbols, pd.date_range(start_date, end_date)) 
        prices = prices.iloc[:,1:]
        prices['Cash'] = 1

        fee = tradings.copy()
        fee.loc[:,:] = 0
        for i in tradings.index:
            for j in symbols:
                if tradings.loc[i, j] != 0:
                    fee.loc[i,j] = commission + prices.loc[i,j]*impact*abs(int(tradings.loc[i,j]))
                else:
                    fee.loc[i,j] = 0

        fees = fee.sum(axis = 1)

        ### II. Update Cash column
        tradings['Cash'] = 0-(prices * tradings).sum(axis = 1) - fees
   
        # (3) get holdings dataframe
        holdings = tradings.copy()
        holdings.loc[:,:] = 0 #initialize all cell to be 0
        holding = holdings.iloc[0,:].copy() #initialize one row in holding

        ### I Update holdings shares
        for date in tradings.index:
            holding += tradings.loc[date,:]
            holdings.loc[date, :] = holding

        ### II. Update holdings cash
        holdings['Cash'] = start_val + holdings['Cash']
        

        ### III. acccount commissions and market impact for cash col

        # (4) get values dataframe
        values = tradings.copy()
        values.iloc[:,:] = 0 # initialize each cell to be 0
    
        values = prices * holdings

        # (5)get portvals with dataframe type
        portvals = pd.DataFrame(values.iloc[:,:-1].sum(axis = 1) + values['Cash'])

        # print(portvals)

        return portvals


    # In the template, instead of computing the value of the portfolio, we just  		  	   		     		  		  		    	 		 		   		 		  
    # read in the value of IBM over 6 months  		  	   		     		  		  		    	 		 		   		 		  
    # start_date = dt.datetime(2008, 1, 1)  		  	   		     		  		  		    	 		 		   		 		  
    # end_date = dt.datetime(2008, 6, 1)  		  	   		     		  		  		    	 		 		   		 		  
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))  		  	   		     		  		  		    	 		 		   		 		  
    # portvals = portvals[["IBM"]]  # remove SPY  		  	   		     		  		  		    	 		 		   		 		  
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)  	
    # # plot_data(portvals)
	   		     		  		  		    	 		 		   		 		  
    # return rv  		  	   		     		  		  		    	 		 		   		 		  
    # return portvals  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def author():
    return 'yjiang347'
	   		     		  		  		    	 		 		   		 		  
def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		     		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		     		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # of = "./orders/orders-02.csv"  
    of = open("./additional_orders/orders2.csv")	
    # of = "./additional_orders/orders2.csv"	  	   		     		  		  		    	 		 		   		 		  
    sv = 1000000  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Process orders  		  	   		     		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		     		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		     		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		     		  		  		    	 		 		   		 		  
    else:  		  	   		     		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Get portfolio stats 
    cum_ret = (portvals[-1]/portvals[0])-1
    daily_ret = (portvals/portvals.shift(1)) - 1
    daily_ret[0] = 0
    daily_ret = daily_ret[1:]
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    sharpe_ratio = (avg_daily_ret/std_daily_ret) * (252**0.5)

    start_date = portvals.index[0] 		  	   		     		  		  		    	 		 		   		 		  
    end_date = portvals.index[-1] 
    spy_data = get_data(['SPY'], pd.date_range(start_date, end_date))
    print(spy_data)
    spy_data = spy_data['SPY']
    

    cum_ret_SPY = (spy_data[-1]/spy_data[0])-1
    daily_ret_SPY = (spy_data/spy_data.shift(1)) - 1
    daily_ret_SPY[0] = 0
    print(daily_ret_SPY)
    daily_ret_SPY = daily_ret_SPY[1:]
    avg_daily_ret_SPY = daily_ret_SPY.mean()
    std_daily_ret_SPY = daily_ret_SPY.std()
    sharpe_ratio_SPY = (avg_daily_ret_SPY/std_daily_ret_SPY) * (252**0.5)
    print("num_days = {}".format(portvals.index.size))

    # plot_data(portvals)

    # Here we just fake the data. you should use your code from previous assignments.  		  	   		     		  		  		    	 		 		   		 		  
    # start_date = dt.datetime(2008, 1, 1)  		  	   		     		  		  		    	 		 		   		 		  
    # end_date = dt.datetime(2008, 6, 1)  		  	   		     		  		  		    	 		 		   		 		  
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		     		  		  		    	 		 		   		 		  
    #     0.2,  		  	   		     		  		  		    	 		 		   		 		  
    #     0.01,  		  	   		     		  		  		    	 		 		   		 		  
    #     0.02,  		  	   		     		  		  		    	 		 		   		 		  
    #     1.5,  		  	   		     		  		  		    	 		 		   		 		  
    # ]  		  	   		     		  		  		    	 		 		   		 		  
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		     		  		  		    	 		 		   		 		  
    #     0.2,  		  	   		     		  		  		    	 		 		   		 		  
    #     0.01,  		  	   		     		  		  		    	 		 		   		 		  
    #     0.02,  		  	   		     		  		  		    	 		 		   		 		  
    #     1.5,  		  	   		     		  		  		    	 		 		   		 		  
    # ]  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    test_code()  		  	   		     		  		  		    	 		 		   		 		  
