
""" 
    Experiment 1: Create chart for in sample JPM (2008-1-1, 2009-12-31):
        - Value of the ManualStrategy portfolio (normalized to 1.0 at the start) -- > red
        - Value of the StrategyLearner portfolio (normalized to 1.0 at the start) --> blue
        - Value of the Benchmark portfolio (normalized to 1.0 at the start) --> green
"""

import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as mk
import matplotlib.pyplot as plt

import datetime as dt
import pandas as pd

from util import get_data

def exp1_plot(symbol = 'JPM', sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), impact=0.005, commission=9.95, sv=100000):
    """
        exp1 creates chart
            - Value of the ManualStrategy portfolio (normalized to 1.0 at the start) -- > red
            - Value of the StrategyLearner portfolio (normalized to 1.0 at the start) --> blue
            - Value of the Benchmark portfolio (normalized to 1.0 at the start) --> green

        :param symbol: symbol of stock
        :typpe symbol: str
        :param sd: start date
        :typpe sd: datetime
        :param ed: end date 
        :typpe ed: datetime
        :param impact: market impact
        :typpe impact: float
        :param commission: each trading cost
        :typpe symbol: float
        :param sv: start value
        :type: float
        :return plot named exp1.png in working dir
        :rtype: png
    """
    #----------------------------------------Generate dataset for ms, sl and benchmark------------------------------------
    # read in price
    price_all = get_data([symbol] ,pd.date_range(sd,ed))
    price = price_all[symbol].to_frame() #only portfolio symbol 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Benchmark~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #create trades df for benchmark
    bench_trades = pd.DataFrame(0, index = price.index, columns = price.columns)
    bench_trades.iloc[0,:] = 1000
    #daily portfolio value 
    bench_portvals = mk.compute_portvals(bench_trades, sv, commission, impact)
    bench_norm_portvals = bench_portvals/bench_portvals.iloc[0,:]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ManualStrategy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #compute trades df
    ms_trades = ms.testPolicy(symbol, sd , ed, sv)
    #compute portfiolio
    ms_portvals =  mk.compute_portvals(ms_trades, sv, commission, impact)
    ms_norm_portvals =  ms_portvals/ms_portvals.iloc[0,:]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~StrategyLearner~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #training
    sl_learner = sl.StrategyLearner(verbose = False, impact = impact, commission= commission)
    sl_learner.add_evidence(symbol, sd, ed, sv)
    #testing
    sl_trades = sl_learner.testPolicy(symbol, sd, ed, sv)
    #compute portfolio
    sl_portvals = mk.compute_portvals(sl_trades, sv, commission, impact)
    sl_norm_portvals = sl_portvals/sl_portvals.iloc[0,:]

    #-----------------------------------------------------PLOTTING--------------------------------------------------------
    fig1 = plt.figure(figsize = (12,8))
    plt.plot(sl_norm_portvals, label = "StrategyLearner", color = 'blue')
    plt.plot(bench_norm_portvals, label = "Benchmark", color = 'green')
    plt.plot(ms_norm_portvals, label = 'ManualStrategy', color = 'red')
    plt.title('Benchmark vs. Strategy Learner vs. Manual Strategy \n(Portfolio Value)')
    plt.ylabel('Normalized Portfolio Value')
    plt.xlabel('Date')
    plt.legend(loc = 'upper left')
    plt.xticks(rotation = 30)
    plt.savefig('exp1.png')
   

#implement author func
def author(self):
    return 'yjiang347'

if __name__ == "__main__":
    exp1_plot()