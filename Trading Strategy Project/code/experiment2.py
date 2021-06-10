""" 
    Experiment 2: StrategyLearner that shows how changing the value of impact should affect in-sample 
    trading behavior for JPM (2008-1-1, 2009-12-31) with three metrics, assess minimum three measurements for
    each.
        - metric_1: cum return
        - metric_2: mean of daily return 
        - metric_3: stdev of daily return
"""

import StrategyLearner as sl
import marketsimcode as mk
import matplotlib.pyplot as plt

import datetime as dt
import pandas as pd
import numpy as np

from util import get_data

def exp2_impactEffect(symbol = 'JPM', sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), impacts=[0.005, 0.01, 0.015], commission=0, sv=100000 ):
    """
        exp2 investigates impact effect on strategy learner in-sample trading behavior
            - metric_1: cum return
            - metric_2: mean of daily return 
            - metric_3: stdev of  daily return

        :param symbol: symbol of stock
        :typpe symbol: str
        :param sd: start date
        :typpe sd: datetime
        :param ed: end date 
        :typpe ed: datetime
        :param impact: list of impact you wanna investigate
        :typpe impact: float
        :param commission: each trading cost
        :typpe symbol: floa
        :param sv: start value
        :type: float
        :return metrics df and plot cum return vs. impact, mean of daily return vs. impact, std of daily return vs. impact and saved as png in working dir
        :rtype: dt and png
    """


    #----------------------------------------Using different impact to evaluate its effects------------------------------------
    # set common parameters
    dates = pd.date_range(sd, ed)

    # read in price
    price_all = get_data([symbol] ,dates)
    price = price_all[symbol].to_frame() #only portfolio symbol

    #update metrics for each impact

    # impacts = [x/1000 for x in range(0, 31, 5)] # set 10 impact
    metrics = np.zeros((len(impacts),4))
    metrics[:,0] = impacts

    for i in range(len(impacts)):
        sl_learner = sl.StrategyLearner(verbose = False, impact = metrics[i,0], commission= commission)
        sl_learner.add_evidence(symbol, sd, ed, sv)
        #testing
        sl_trades = sl_learner.testPolicy(symbol, sd, ed, sv)
        # print(sl_trades)
        #compute portfolio
        sl_portvals = mk.compute_portvals(sl_trades, sv, commission, metrics[i,0])
        # print(sl_portvals)

        #cumpute metrics
        cum_return = (sl_portvals.iloc[-1]/sl_portvals.iloc[0]) -1

        daily_ret = (sl_portvals/sl_portvals.shift(1)) -1
        # print (daily_ret)
        daily_ret.iloc[0] = 0
        daily_ret = daily_ret.iloc[1:]

        mean_daily_ret = daily_ret.mean()
        std_daily_ret = daily_ret.std()

        #update metrics ndarray
        metrics[i,1] = cum_return.iloc[0]
        metrics[i,2] = mean_daily_ret
        metrics[i,3] = std_daily_ret

    colors = ["blue", "pink", 'green', "orange"]
    labels = ["impact", "Cumulative Return", "Mean of Daily Return", "Std of Daily Return"]

    metrics_df = pd.DataFrame(metrics[:,1:], index = metrics[:,0], columns = labels[1:])

    # metrics_df.to_csv("exp2-Metrics.csv")

    # print(f"Metrics in different impacts:\n{metrics_df}")

    for i in range(1, metrics.shape[1]):
        fig = plt.figure(figsize = (10,8))
        plt.plot(metrics[:,0], metrics[:,i], color = colors[i])
        plt.title(f'{labels[i]} of Strategy Learner \n with Different Impacts')
        plt.ylabel(f'{labels[i]}')
        plt.xlabel('Impact')
        # plt.legend(loc = 'lower right')
        # plt.xticks(rotation = 30)
        plt.savefig(f'exp2-{i}.png')

    # return metrics_df

#implement author func
def author(self):
    return 'yjiang347'