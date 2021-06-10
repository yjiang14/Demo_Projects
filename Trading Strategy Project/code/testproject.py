"""This file is used to generate plots, results showed in this project report"""

import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as mk
import experiment1 as exp1
import experiment2 as exp2
import matplotlib.pyplot as plt

import datetime as dt
import pandas as pd

from util import get_data

#---------------------------------------------------Common Params----------------------------------------------
symbol = "JPM"
sd_in = dt.datetime(2008,1,1)
ed_in = dt.datetime(2009,12,31)
sd_out = dt.datetime(2010,1,1)
ed_out = dt.datetime(2011,12,31)
sv = 100000
commission = 9.95
impact = 0.005
#-----------------------------------------------------Manual Strategy-------------------------------------------
# read in price
price_all = get_data([symbol] ,pd.date_range(sd_in, ed_in))
price = price_all[symbol].to_frame() #only portfolio symbol 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~chart 1: in-sample~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Benchmark~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#create trades df for benchmark
bench_trades = pd.DataFrame(0, index = price.index, columns = price.columns)
bench_trades.iloc[0,:] = 1000
#daily portfolio value 
bench_portvals = mk.compute_portvals(bench_trades, sv, commission, impact)
bench_norm_portvals = bench_portvals/bench_portvals.iloc[0,:]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ManualStrategy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#compute trades df
ms_trades = ms.testPolicy(symbol, sd_in , ed_in, sv)

#compute portfiolio
ms_portvals =  mk.compute_portvals(ms_trades, sv, commission, impact)
ms_norm_portvals =  ms_portvals/ms_portvals.iloc[0,:]
#initialized trades df
ms_trades = ms.testPolicy(symbol, sd_in , ed_in, sv)
ms_trades_long = ms_trades[ms_trades > 0].dropna()
ms_trades_short = ms_trades[ms_trades < 0].dropna()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig5 = plt.figure(figsize = (12,8))
line1 = plt.plot(ms_norm_portvals, label = "ManualStrategy", color = 'red')
line2 = plt.plot(bench_norm_portvals, label = "Benchmark", color = 'green')
ylim = list(plt.gca().get_ylim())
line3 = plt.vlines (ms_trades_long.index, ymin = ylim[0] , ymax = ylim[1], color = 'blue', label = 'Long Entry Point')
line4 = plt.vlines (ms_trades_short.index, ymin = ylim[0] , ymax = ylim[1], color = 'black', label = 'Short Entry Point')
plt.title('Manual Strategy vs. Benchmark\n(In-Sample Portfolio Value)')
plt.ylabel('Normalized Portfolio Value')
plt.xlabel('Date')
plt.legend(loc = 'upper left')
# plt.legend(handles = [line3, line4],labels = ["Long", "Short"], bbox_to_anchor=(1.1, 1),loc = 'upper right')
plt.xticks(rotation = 30)
plt.savefig('ms_in.png')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~chart 2: out-of-sample~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read in price
price_all_out = get_data([symbol] ,pd.date_range(sd_out, ed_out))
price_out = price_all_out[symbol].to_frame() #only portfolio symbol 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Benchmark~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#create trades df for benchmark
bench_trades_out = pd.DataFrame(0, index = price_out.index, columns = price_out.columns)
bench_trades_out.iloc[0,:] = 1000
#daily portfolio value 
bench_portvals_out = mk.compute_portvals(bench_trades_out, sv, commission, impact)
bench_norm_portvals_out = bench_portvals_out/bench_portvals_out.iloc[0,:]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ManualStrategy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#compute trades df
ms_trades_out = ms.testPolicy(symbol, sd_out , ed_out, sv)

#compute portfiolio
ms_portvals_out =  mk.compute_portvals(ms_trades_out, sv, commission, impact)
ms_norm_portvals_out =  ms_portvals_out/ms_portvals_out.iloc[0,:]
#initialized trades df
ms_trades_out = ms.testPolicy(symbol, sd_out , ed_out, sv)
ms_trades_long_out = ms_trades_out[ms_trades_out > 0].dropna()
ms_trades_short_out = ms_trades_out[ms_trades_out < 0].dropna()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig6 = plt.figure(figsize = (12,8))
line5 = plt.plot(ms_norm_portvals_out, label = "ManualStrategy", color = 'red')
line6 = plt.plot(bench_norm_portvals_out, label = "Benchmark", color = 'green')
ylim_out = list(plt.gca().get_ylim())
line7 = plt.vlines (ms_trades_long_out.index, ymin = ylim_out[0] , ymax = ylim_out[1], color = 'blue', label = 'Long Entry Point')
line8 = plt.vlines (ms_trades_short_out.index, ymin = ylim_out[0] , ymax = ylim_out[1], color = 'black', label = 'Short Entry Point')
plt.title('Manual Strategy vs. BenchMark\n(Out-of-Sample Portfolio Value)')
plt.ylabel('Normalized Portfolio Value')
plt.xlabel('Date')
plt.legend(loc = 'upper left')
# plt.legend(handles = [line3, line4],labels = ["Long", "Short"], bbox_to_anchor=(1.1, 1),loc = 'upper right')
plt.xticks(rotation = 30)
plt.savefig('ms_out.png')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Summarize Table~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sum_table = pd.DataFrame(0, index = ["Benchmark", "Manual Strategy"], 
                        columns = ["In-Cumulative Return", "Out-Cumulative Return", "In-Mean of Daily Return", "Out-Mean of Daily Return",
                        "In-STDEV of Daily Return","Out-STDEV of Daily Return"])
#Benchmark
# in-sample
bench_cumret_in = (bench_portvals.iloc[-1]/bench_portvals.iloc[0]) -1
bench_dailyret_in = (bench_portvals/bench_portvals.shift(1)) -1
bench_dailyret_in.iloc[0] = 0
bench_dailyret_in = bench_dailyret_in.iloc[1:] 
bench_meandr_in = bench_dailyret_in.mean()
bench_stddr_in = bench_dailyret_in.std()
#out-of-sample
bench_cumret_out = (bench_portvals_out.iloc[-1]/bench_portvals_out.iloc[0]) -1
bench_dailyret_out = (bench_portvals_out/bench_portvals_out.shift(1)) -1
bench_dailyret_out.iloc[0] = 0
bench_dailyret_out = bench_dailyret_out.iloc[1:]
bench_meandr_out = bench_dailyret_out.mean()
bench_stddr_out = bench_dailyret_out.std()

#ManualStrategy
# in-sample
ms_cumret_in = (ms_portvals.iloc[-1]/ms_portvals.iloc[0]) -1
ms_dailyret_in = (ms_portvals/ms_portvals.shift(1)) -1
ms_dailyret_in.iloc[0] = 0
ms_dailyret_in = ms_dailyret_in.iloc[1:]
ms_meandr_in = ms_dailyret_in.mean()
ms_stddr_in = ms_dailyret_in.std()
# out-of-sample
ms_cumret_out = (ms_portvals_out.iloc[-1]/ms_portvals_out.iloc[0]) -1
ms_dailyret_out = (ms_portvals_out/ms_portvals_out.shift(1)) -1
ms_dailyret_out.iloc[0] = 0
ms_dailyret_out = ms_dailyret_out.iloc[1:]
ms_meandr_out = ms_dailyret_out.mean()
ms_stddr_out = ms_dailyret_out.std()

#update sum_table
sum_table['In-Cumulative Return'] = [bench_cumret_in.iloc[0], ms_cumret_in.iloc[0]]
sum_table['Out-Cumulative Return'] = [bench_cumret_out.iloc[0], ms_cumret_out.iloc[0]]
sum_table['In-Mean of Daily Return'] = [bench_meandr_in.iloc[0], ms_meandr_in.iloc[0]]
sum_table['Out-Mean of Daily Return'] = [bench_meandr_out.iloc[0], ms_meandr_out.iloc[0]]
sum_table['In-STDEV of Daily Return'] = [bench_stddr_in.iloc[0], ms_stddr_in.iloc[0]]
sum_table['Out-STDEV of Daily Return'] = [bench_stddr_out.iloc[0], ms_stddr_out.iloc[0]]

print(f"The Summary of Performance:\n {sum_table}")
sum_table.to_csv("ms_vs_benchmark.csv")

# sum_table_norm = sum_table/sum_table.iloc[0,:]
# fig7 = plt.figure(figsize=(8,8))
# plt.plot(sum_table_norm.iloc[:,0], label = "In-sample Cumulative Return", color = 'lime', marker = '*')
# plt.plot(sum_table_norm.iloc[:,1], label = "Out-of-Sample Cumulative Return", color = 'magenta', marker = '*')
# plt.plot(sum_table_norm.iloc[:,2], label = "In-sample Mean of Daily Return", color = 'salmon', linestyle = '--', marker = '.')
# plt.plot(sum_table_norm.iloc[:,3], label = "Out-of-sample Mean of Daily Return", color = 'grey', linestyle = '--', marker = '.')
# plt.plot(sum_table_norm.iloc[:,4], label = "In-sample STDEV of Daily Return", color = 'olive', linestyle = ':', marker = 'X')
# plt.plot(sum_table_norm.iloc[:,5], label = "Out-of-sample STDEV of Daily Return", color = 'Orchid', linestyle = ':', marker = 'X')
# plt.title('Benchmark Perfomance vs. Manual Strategy Performace')
# plt.ylabel('Metric_Value')
# plt.xlabel('Approach')
# plt.legend(loc = 'upper left')
# plt.legend(handles = [line3, line4],labels = ["Long", "Short"], bbox_to_anchor=(1.1, 1),loc = 'upper right')
# plt.xticks(rotation = 30)
# plt.savefig('Figures/ms_out.png')

#~~~~~~~~~~~~~~~~~~~~~~~~~~the whole plot for performance in sample vs. out sample~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig7, axs1= plt.subplots(1,3, figsize = (21,6))
fig7.suptitle("Manual Strategy Performance vs. Benchmark Performance")
# fig7.subplots_adjust(hspace = 0.15)
# plt.setp(axs1[0].get_xticklabels())


#-------------------subplot1: cum_ret---------------------------
axs1[0].plot(sum_table.iloc[:,0], label = "In-sample Cumulative Return", color = 'teal', marker = 'X',linestyle = '--')
axs1[0].plot(sum_table.iloc[:,1], label = "Out-of-Sample Cumulative Return", color = 'magenta', marker = 'X',linestyle = '--')
# axs1[0].set_xlim(('2008-01-01', '2010-01-01'))
# axs1[0].set_ylim((0,2))
axs1[0].legend(loc = 'upper left')
axs1[0].grid(color='gray', ls = '-.', lw = 0.25)
axs1[0].set_title('Manual Strategy vs, Benchmark\nin Cumulative Return')
axs1[0].set_ylabel('Cumulative Return')
# axs1[0].tick_params(axis = 'x', bottom = False)


# #---------------------subplot2: mean_dailyret-----------------------
axs1[1].plot(sum_table.iloc[:,2], label = "In-sample Mean of Daily Return", color = 'teal', linestyle = '--', marker = 'X')
axs1[1].plot(sum_table.iloc[:,3], label = "Out-of-sample Mean of Daily Return", color = 'magenta', linestyle = '--', marker = 'X')
axs1[1].set_title('Manual Strategy vs, Benchmark\nin Mean of Daily Return')
# axs1[1].set_title('Volume vs. Date')
axs1[1].grid(color='gray', ls = '-.', lw = 0.25)
axs1[1].set_ylabel('Mean of Daily Return')
# axs1[1].tick_params(axis = 'x', bottom = False)
axs1[1].legend(loc = 'upper left')

#---------------------subplot3: std_dailyret---------------------------
axs1[2].plot(sum_table.iloc[:,4], label = "In-sample STDEV of Daily Return", color = 'teal', linestyle = '--', marker = 'X')
axs1[2].plot(sum_table.iloc[:,5], label = "Out-of-sample STDEV of Daily Return", color = 'magenta', linestyle = '--', marker = 'X')
axs1[2].set_title('Manual Strategy vs, Benchmark\nin STDEV of Daily Return')
# axs1[2].set_title('Volume vs. Date')
axs1[2].grid(color='gray', ls = '-.', lw = 0.25)
axs1[2].set_ylabel('STDEV of Daily Return')
# axs1[2].tick_params(axis = 'x', bottom = False)
axs1[2].legend(loc = 'upper right')
plt.savefig('performance.png')

#--------------------------------------------------- Strategy Learner-------------------------------------------

#-----------------------------------------------------EXP1: Plot------------------------------------------------------
exp1.exp1_plot(symbol = 'JPM', sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), impact=0.005, commission=0, sv=100000)
# -----------------------------------------------------EXP2: plot------------------------------------------------------
impacts = [x/1000 for x in range(0, 31, 5)]
exp2.exp2_impactEffect(symbol = 'JPM', sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), impacts = impacts, commission=0, sv=100000 )

def author():
    return 'yjiang347'