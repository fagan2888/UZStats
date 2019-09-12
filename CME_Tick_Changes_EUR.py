# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The Robert and Rosenbaum Uncertainty Zones model

# %% [markdown]
# # An application to EURUSD FX Futures at CME

# %% [markdown]
# ## Implementation by
# ## Marcos Costa Santos Carreira (École Polytechnique - CMAP)
# ## and
# ## Florian Huchedé (CME)
# ## Aug-2019

# %% [markdown]
# ## Import packages

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import glob

# %%
pd.set_option('display.max_columns', 50)

# %%
pd.set_option('display.max_rows', 200)

# %%
import cme_processing as cme

# %% [markdown]
# ## File paths and initial values

# %%
PATHPROJ = '/Users/marcoscscarreira/Documents/X/CME project/CME_data/'
URL_ROOT = 'https://raw.githubusercontent.com/MarcosCarreira/UZStats/master/CME_data/'

# %%
CURR = 'EUR'

# %%
PATH_PRIOR = PATHPROJ+CURR+'/prior/'
PATH_AFTER = PATHPROJ+CURR+'/after/'
URL_1 = CURR+'/prior/'
URL_2 = CURR+'/after/'
#PATH_PRIOR = URL_ROOT+URL_1
#PATH_AFTER = URL_ROOT+URL_2

# %%
TRADING_HOURS = 8

# %%
TICK_PRIOR = 1.0
TICK_AFTER = 0.5

# %%
PRIOR_CDATES_LIST = [['6EU5', '20150615'], ['6EU5', '20150616'], ['6EU5', '20150617'],\
    ['6EU5', '20150618'], ['6EU5', '20150619'], ['6EU5', '20150622'], ['6EU5', '20150623'],\
    ['6EU5', '20150624'], ['6EU5', '20150625'], ['6EU5', '20150626'], ['6EU5', '20150629'],\
    ['6EU5', '20150630'], ['6EU5', '20150701'], ['6EU5', '20150702'], ['6EU5', '20150703'],\
    ['6EU5', '20150706'], ['6EU5', '20150707'], ['6EU5', '20150708'], ['6EU5', '20150709'],\
    ['6EU5', '20150710'], ['6EU5', '20150713'], ['6EU5', '20150714'], ['6EU5', '20150715'],\
    ['6EU5', '20150716'], ['6EU5', '20150717'], ['6EU5', '20150720'], ['6EU5', '20150721'],\
    ['6EU5', '20150722'], ['6EU5', '20150723'], ['6EU5', '20150724'], ['6EU5', '20150727'],\
    ['6EU5', '20150728'], ['6EU5', '20150729'], ['6EU5', '20150730'], ['6EU5', '20150731'],\
    ['6EU5', '20150803'], ['6EU5', '20150804'], ['6EU5', '20150805'], ['6EU5', '20150806'],\
    ['6EU5', '20150807'], ['6EU5', '20150810'], ['6EU5', '20150811'], ['6EU5', '20150812'],\
    ['6EU5', '20150813'], ['6EU5', '20150814'], ['6EU5', '20150817'], ['6EU5', '20150818'],\
    ['6EU5', '20150819'], ['6EU5', '20150820'], ['6EU5', '20150821'], ['6EU5', '20150824'],\
    ['6EU5', '20150825'], ['6EU5', '20150826'], ['6EU5', '20150827'], ['6EU5', '20150828'],\
    ['6EU5', '20150831'], ['6EU5', '20150901'], ['6EU5', '20150902'], ['6EU5', '20150903'],\
    ['6EU5', '20150904'], ['6EU5', '20150907'], ['6EU5', '20150908'], ['6EU5', '20150909'],\
    ['6EU5', '20150910'], ['6EU5', '20150911'], ['6EZ5', '20150914'], ['6EZ5', '20150915'],\
    ['6EZ5', '20150916'], ['6EZ5', '20150917'], ['6EZ5', '20150918'], ['6EZ5', '20150921'],\
    ['6EZ5', '20150922'], ['6EZ5', '20150923'], ['6EZ5', '20150924'], ['6EZ5', '20150925'],\
    ['6EZ5', '20150928'], ['6EZ5', '20150929'], ['6EZ5', '20150930'], ['6EZ5', '20151001'],\
    ['6EZ5', '20151002'], ['6EZ5', '20151005'], ['6EZ5', '20151006'], ['6EZ5', '20151007'],\
    ['6EZ5', '20151008'], ['6EZ5', '20151009'], ['6EZ5', '20151012'], ['6EZ5', '20151013'],\
    ['6EZ5', '20151014'], ['6EZ5', '20151015'], ['6EZ5', '20151016'], ['6EZ5', '20151019'],\
    ['6EZ5', '20151020'], ['6EZ5', '20151021'], ['6EZ5', '20151022'], ['6EZ5', '20151023'],\
    ['6EZ5', '20151026'], ['6EZ5', '20151027'], ['6EZ5', '20151028'], ['6EZ5', '20151029'],\
    ['6EZ5', '20151030'], ['6EZ5', '20151102'], ['6EZ5', '20151103'], ['6EZ5', '20151104'],\
    ['6EZ5', '20151105'], ['6EZ5', '20151106'], ['6EZ5', '20151109'], ['6EZ5', '20151110'],\
    ['6EZ5', '20151111'], ['6EZ5', '20151112'], ['6EZ5', '20151113'], ['6EZ5', '20151116'],\
    ['6EZ5', '20151117'], ['6EZ5', '20151118'], ['6EZ5', '20151119'], ['6EZ5', '20151120'],\
    ['6EZ5', '20151123'], ['6EZ5', '20151124'], ['6EZ5', '20151125'], ['6EZ5', '20151126'],\
    ['6EZ5', '20151127'], ['6EZ5', '20151130'], ['6EZ5', '20151201'], ['6EZ5', '20151202'],\
    ['6EZ5', '20151203'], ['6EZ5', '20151204'], ['6EZ5', '20151207'], ['6EZ5', '20151208'],\
    ['6EZ5', '20151209'], ['6EZ5', '20151210'], ['6EZ5', '20151211'], ['x6EH6', '20151214'],\
    ['x6EH6', '20151215'], ['x6EH6', '20151216'], ['x6EH6', '20151217'], ['x6EH6', '20151218'],\
    ['x6EH6', '20151221'], ['x6EH6', '20151222'], ['x6EH6', '20151223'], ['x6EH6', '20160104'],\
    ['x6EH6', '20160105'], ['x6EH6', '20160106'], ['x6EH6', '20160107'], ['x6EH6', '20160108']]

# %%
AFTER_CDATES_LIST = [['x6EH6', '20160111'], ['x6EH6', '20160112'], ['x6EH6', '20160113'],\
    ['x6EH6', '20160114'], ['x6EH6', '20160115'], ['x6EH6', '20160118'], ['x6EH6', '20160119'],\
    ['x6EH6', '20160120'], ['x6EH6', '20160121'], ['x6EH6', '20160122'], ['x6EH6', '20160125'],\
    ['x6EH6', '20160126'], ['x6EH6', '20160127'], ['x6EH6', '20160128'], ['x6EH6', '20160129'],\
    ['x6EH6', '20160201'], ['x6EH6', '20160202'], ['x6EH6', '20160203'], ['x6EH6', '20160204'],\
    ['x6EH6', '20160205'], ['x6EH6', '20160208'], ['x6EH6', '20160209'], ['x6EH6', '20160210'],\
    ['x6EH6', '20160211'], ['x6EH6', '20160212'], ['x6EH6', '20160215'], ['x6EH6', '20160216'],\
    ['x6EH6', '20160217'], ['x6EH6', '20160218'], ['x6EH6', '20160219'], ['x6EH6', '20160222'],\
    ['x6EH6', '20160223'], ['x6EH6', '20160224'], ['x6EH6', '20160225'], ['x6EH6', '20160226'],\
    ['x6EH6', '20160229'], ['x6EH6', '20160301'], ['x6EH6', '20160302'], ['x6EH6', '20160303'],\
    ['x6EH6', '20160304'], ['x6EH6', '20160307'], ['x6EH6', '20160308'], ['x6EH6', '20160309'],\
    ['x6EH6', '20160310'], ['x6EH6', '20160311'], ['x6EM6', '20160314'], ['x6EM6', '20160315'],\
    ['x6EM6', '20160316'], ['x6EM6', '20160317'], ['x6EM6', '20160318'], ['x6EM6', '20160321'],\
    ['x6EM6', '20160322'], ['x6EM6', '20160323'], ['x6EM6', '20160324'], ['x6EM6', '20160328'],\
    ['x6EM6', '20160329'], ['x6EM6', '20160330'], ['x6EM6', '20160331'], ['x6EM6', '20160401'],\
    ['x6EM6', '20160404'], ['x6EM6', '20160405'], ['x6EM6', '20160406'], ['x6EM6', '20160407'],\
    ['x6EM6', '20160408'], ['x6EM6', '20160411'], ['x6EM6', '20160412'], ['x6EM6', '20160413'],\
    ['x6EM6', '20160414'], ['x6EM6', '20160415'], ['x6EM6', '20160418'], ['x6EM6', '20160419'],\
    ['x6EM6', '20160420'], ['x6EM6', '20160421'], ['x6EM6', '20160422'], ['x6EM6', '20160425'],\
    ['x6EM6', '20160426'], ['x6EM6', '20160427'], ['x6EM6', '20160428'], ['x6EM6', '20160429'],\
    ['x6EM6', '20160502'], ['x6EM6', '20160503'], ['x6EM6', '20160504'], ['x6EM6', '20160505'],\
    ['x6EM6', '20160506'], ['x6EM6', '20160509'], ['x6EM6', '20160510'], ['x6EM6', '20160511'],\
    ['x6EM6', '20160512'], ['x6EM6', '20160513'], ['x6EM6', '20160516'], ['x6EM6', '20160517'],\
    ['x6EM6', '20160518'], ['x6EM6', '20160519'], ['x6EM6', '20160520'], ['x6EM6', '20160523'],\
    ['x6EM6', '20160524'], ['x6EM6', '20160525'], ['x6EM6', '20160526'], ['x6EM6', '20160527'],\
    ['x6EM6', '20160530'], ['x6EM6', '20160531'], ['x6EM6', '20160601'], ['x6EM6', '20160602'],\
    ['x6EM6', '20160603'], ['x6EM6', '20160606'], ['x6EM6', '20160607'], ['x6EM6', '20160608'],\
    ['x6EM6', '20160609'], ['x6EM6', '20160610']]

# %% [markdown]
# ### Processing files

# %% [markdown]
# #### Prior

# %%
#PRIOR_CDATES_LIST = cme.list_files(PATH_PRIOR)

# %%
#PRIOR_CDATES_LIST

# %%
PRIOR_CDATES, FILES_PRIOR_CAticks, FILES_PRIOR_COSTtrades,\
    FILES_PRIOR_OBstats, FILES_PRIOR_OTtrans,\
    FILES_PRIOR_RDFtrans, FILES_PRIOR_UZstats = \
    cme.process_files(PATH_PRIOR, PRIOR_CDATES_LIST, 'prior', TICK_PRIOR)

# %%
PRIOR_OB_UZ_STATS = cme.ob_uz_stats(PRIOR_CDATES, FILES_PRIOR_OBstats,\
    FILES_PRIOR_UZstats, FILES_PRIOR_CAticks, TRADING_HOURS)

# %%
PRIOR_IMBAL_STATS = cme.imbal_stats(PRIOR_CDATES, FILES_PRIOR_OTtrans)

# %%
PRIOR_IMBAL_STATS_TS = cme.time_series_imbal(PRIOR_IMBAL_STATS, pd.to_datetime(PRIOR_CDATES['Date']), 'prior')

# %%
PRIOR_IMBAL_STATS_TS['eta1'] = PRIOR_OB_UZ_STATS['eta1'].values

# %%
PRIOR_TRADE_STATS_TS = cme.time_series_imbal_trd(PRIOR_IMBAL_STATS, pd.to_datetime(PRIOR_CDATES['Date']), 'prior')

# %%
PRIOR_DEPL_STATS = cme.depl_stats(PRIOR_CDATES, FILES_PRIOR_RDFtrans)

# %%
PRIOR_DEPL_STATS_TS = cme.time_series_depl(PRIOR_DEPL_STATS, pd.to_datetime(PRIOR_CDATES['Date']), 'prior')

# %%
PRIOR_DEPL_STATS_TS['eta1'] = PRIOR_OB_UZ_STATS['eta1'].values

# %%
PRIOR_ABSDEPL_STATS_TS = cme.time_series_absdepl(PRIOR_DEPL_STATS, pd.to_datetime(PRIOR_CDATES['Date']), 'prior')

# %%
PRIOR_ABSDEPL_STATS_TS['eta1'] = PRIOR_OB_UZ_STATS['eta1'].values
PRIOR_ABSDEPL_STATS_TS['M'] = PRIOR_OB_UZ_STATS['M'].values

# %%
PRIOR_COST_STATS = cme.cost_stats(PRIOR_CDATES, FILES_PRIOR_COSTtrades)

# %%
PRIOR_COST_STATS['Status'] = 'prior'

# %% [markdown]
# #### After

# %%
#AFTER_CDATES_LIST = cme.list_files(PATH_AFTER)

# %%
#AFTER_CDATES_LIST

# %%
AFTER_CDATES, FILES_AFTER_CAticks, FILES_AFTER_COSTtrades,\
    FILES_AFTER_OBstats, FILES_AFTER_OTtrans,\
    FILES_AFTER_RDFtrans, FILES_AFTER_UZstats = \
    cme.process_files(PATH_AFTER, AFTER_CDATES_LIST, 'after', TICK_AFTER)

# %%
AFTER_OB_UZ_STATS = cme.ob_uz_stats(AFTER_CDATES, FILES_AFTER_OBstats,\
    FILES_AFTER_UZstats, FILES_AFTER_CAticks, TRADING_HOURS)

# %%
AFTER_IMBAL_STATS = cme.imbal_stats(AFTER_CDATES, FILES_AFTER_OTtrans)

# %%
AFTER_IMBAL_STATS_TS = cme.time_series_imbal(AFTER_IMBAL_STATS, pd.to_datetime(AFTER_CDATES['Date']), 'after')

# %%
AFTER_IMBAL_STATS_TS['eta1'] = AFTER_OB_UZ_STATS['eta1'].values

# %%
AFTER_TRADE_STATS_TS = cme.time_series_imbal_trd(AFTER_IMBAL_STATS, pd.to_datetime(AFTER_CDATES['Date']), 'after')

# %%
AFTER_DEPL_STATS = cme.depl_stats(AFTER_CDATES, FILES_AFTER_RDFtrans)

# %%
AFTER_DEPL_STATS_TS = cme.time_series_depl(AFTER_DEPL_STATS, pd.to_datetime(AFTER_CDATES['Date']), 'after')

# %%
AFTER_DEPL_STATS_TS['eta1'] = AFTER_OB_UZ_STATS['eta1'].values

# %%
AFTER_ABSDEPL_STATS_TS = cme.time_series_absdepl(AFTER_DEPL_STATS, pd.to_datetime(AFTER_CDATES['Date']), 'after')

# %%
AFTER_ABSDEPL_STATS_TS['eta1'] = AFTER_OB_UZ_STATS['eta1'].values
AFTER_ABSDEPL_STATS_TS['M'] = AFTER_OB_UZ_STATS['M'].values

# %%
AFTER_COST_STATS = cme.cost_stats(AFTER_CDATES, FILES_AFTER_COSTtrades)

# %%
AFTER_COST_STATS['Status'] = 'after'

# %% [markdown]
# #### Join prior and after

# %%
OB_UZ_STATS = pd.concat([PRIOR_OB_UZ_STATS, AFTER_OB_UZ_STATS], sort=False)

# %%
IMBAL_STATS_TS = pd.concat([PRIOR_IMBAL_STATS_TS, AFTER_IMBAL_STATS_TS], sort=False)

# %%
TRADE_STATS_TS = pd.concat([PRIOR_TRADE_STATS_TS, AFTER_TRADE_STATS_TS], sort=False)

# %%
DEPL_STATS_TS = pd.concat([PRIOR_DEPL_STATS_TS, AFTER_DEPL_STATS_TS], sort=False)

# %%
ABSDEPL_STATS_TS = pd.concat([PRIOR_ABSDEPL_STATS_TS, AFTER_ABSDEPL_STATS_TS], sort=False)

# %% [markdown]
# ### Tables

# %%
TABLE_MATHIEU = cme.table_mathieu(OB_UZ_STATS)
TABLE_MATHIEU_ERR = cme.table_mathieu_err(OB_UZ_STATS)

# %%
TABLE_MATHIEU

# %%
TABLE_MATHIEU_ERR

# %%
cme.avg_perc_mat(PRIOR_IMBAL_STATS, pd.to_datetime(PRIOR_CDATES['Date']))

# %%
cme.avg_perc_mat(AFTER_IMBAL_STATS, pd.to_datetime(AFTER_CDATES['Date']))

# %%
cme.avg_perc_mat_2(PRIOR_DEPL_STATS, pd.to_datetime(PRIOR_CDATES['Date']))

# %%
cme.avg_perc_mat_2(AFTER_DEPL_STATS, pd.to_datetime(AFTER_CDATES['Date']))

# %% [markdown]
# ## Charts and Regressions

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='Pred_Imbal_Relat', hue='Status',\
           data=IMBAL_STATS_TS);
plt.title('Relative predictive power of imbalance and $\eta$ : '+CURR);

# %%
cme.time_series_hist_plot(IMBAL_STATS_TS, 'Pred_Imbal_Relat',\
    'Relative predictive power of imbalance : '+CURR, -1.0, 4.0, 50)

# %%
cme.time_series_hist_plot(ABSDEPL_STATS_TS, 'Depl_Cancel',\
    'Depl_Cancel : '+CURR, 0.0, 50000.0, 50)

# %%
cme.time_series_hist_plot(ABSDEPL_STATS_TS, 'Depl_Trades',\
    'Depl_Trades : '+CURR, 0.0, 70000.0, 50)

# %%
cme.regr_plot(ABSDEPL_STATS_TS, 'M', 'Depl_Cancel',\
    'Depletions on Cancels (y) x Number of Trades (x) : '+CURR)

# %%
cme.regr_plot(ABSDEPL_STATS_TS, 'M', 'Depl_Trades',\
    'Depletions on Trades (y) x Number of Trades (x) : '+CURR)

# %%
cme.lin_reg(ABSDEPL_STATS_TS, ['M'], 'Depl_Cancel')

# %%
cme.lin_reg(ABSDEPL_STATS_TS, ['M'], 'Depl_Trades')

# %%
cme.lin_reg(PRIOR_ABSDEPL_STATS_TS, ['M'], 'Depl_Trades')

# %%
cme.lin_reg(AFTER_ABSDEPL_STATS_TS, ['M'], 'Depl_Trades')

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Depl_Cancel',\
    'Depl_Cancel : '+CURR, 0, 45, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DC same-oppo',\
    'DC same-oppo : '+CURR, 0, 30, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Depl_Trade',\
    'Depl_Trade : '+CURR, 0, 40, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DT same-oppo',\
    'DT same-oppo : '+CURR, -2, 20, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DT+F same-oppo',\
    'DT+F same-oppo : '+CURR, -1, 6, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Fill same-oppo',\
    'Filled : Same - Opposite : '+CURR, -15, 25, 50)

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='DC same-oppo', hue='Status',\
           data=DEPL_STATS_TS);
plt.title('DC same-oppo and $\eta$ : '+CURR);

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='DT same-oppo', hue='Status',\
           data=DEPL_STATS_TS);
plt.title('DT same-oppo and $\eta$ : '+CURR);

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='DT+F same-oppo', hue='Status',\
           data=DEPL_STATS_TS);
plt.title('DT+F same-oppo and $\eta$ : '+CURR);

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='Fill same-oppo', hue='Status',\
           data=DEPL_STATS_TS);
plt.title('Fill and $\eta$ : '+CURR);

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'twspr1',\
    'Spread in Ticks : '+CURR, 1, 2, 50)

# %%
cme.twspr_plot_USD(OB_UZ_STATS, CURR)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'eta1',\
    '$\eta$ : '+CURR, 0, 0.5, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'chgavg',\
    'Average Price Change : '+CURR, 0.4, 1.4, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'rvxe',\
    'Estimated Volatility of Efficient Prices : '+CURR, 0, 0.015, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'ndfpr',\
    'Number of Price Changes : '+CURR, 0, 60000, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'M',\
    'Number of Trades : '+CURR, 0, 150000, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'Volume',\
    'Volume : '+CURR, 0, 500000, 50)

# %%
cme.scatter_plot(OB_UZ_STATS, 'chgavg', 'eta1',\
    'Eta (y) x Average Price Change (x) : '+CURR)

# %%
cme.scatter_plot(OB_UZ_STATS, 'rvxe', 'twspr1',\
    'Spread in Ticks (y) x Estimated Volatility of Efficient Prices (x) : '+CURR)

# %%
cme.scatter_plot(OB_UZ_STATS, 'eta1', 'twspr1',\
    'Spread in Ticks (y) x Eta (x) : '+CURR)

# %%
cme.time_series_hist(OB_UZ_STATS, 'eta1',\
    'Eta Histogram : '+CURR)

# %%
cme.scatter_plot(OB_UZ_STATS, 'duration', 'dt_avg',\
    'Realized Duration (y) x Predicted Duration (x) : '+CURR)

# %%
cme.scatter_plot(OB_UZ_STATS, 'twspr1', 'eta1',\
    'Eta (y) x Time-Weighted Average Spread (x) : '+CURR)

# %%
cme.cloud1(OB_UZ_STATS, CURR)

# %%
cme.cloud1(OB_UZ_STATS, CURR, True)

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS, ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')

# %%
cme.lin_reg_rob(PRIOR_OB_UZ_STATS, ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')

# %%
cme.lin_reg(AFTER_OB_UZ_STATS, ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')

# %%
cme.lin_reg_rob(AFTER_OB_UZ_STATS, ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')

# %%
OB_UZ_STATS['p1*eta*alpha*sqrt(M)'] = np.where(OB_UZ_STATS['Status']=='prior',\
    cme.lin_reg_params(PRIOR_OB_UZ_STATS, ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')['eta*alpha*sqrt(M)'],\
    cme.lin_reg_params(AFTER_OB_UZ_STATS, ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')['eta*alpha*sqrt(M)'])\
    *OB_UZ_STATS['eta*alpha*sqrt(M)']
OB_UZ_STATS['sigma-p2*S*sqrt(M)'] = OB_UZ_STATS['sigma']-\
    np.where(OB_UZ_STATS['Status']=='prior',\
    cme.lin_reg_params(PRIOR_OB_UZ_STATS, ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')['S*sqrt(M)'],\
    cme.lin_reg_params(AFTER_OB_UZ_STATS, ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')['S*sqrt(M)'])*\
    OB_UZ_STATS['S*sqrt(M)']

# %%
cme.cloud2(OB_UZ_STATS, CURR)

# %%
cme.cloud2(OB_UZ_STATS, CURR, True)

# %%
cme.lin_reg(OB_UZ_STATS[OB_UZ_STATS['Status']=='prior'], ['p1*eta*alpha*sqrt(M)'], 'sigma-p2*S*sqrt(M)')

# %%
cme.lin_reg_rob(OB_UZ_STATS[OB_UZ_STATS['Status']=='prior'], ['p1*eta*alpha*sqrt(M)'], 'sigma-p2*S*sqrt(M)')

# %%
cme.lin_reg(OB_UZ_STATS[OB_UZ_STATS['Status']=='after'], ['p1*eta*alpha*sqrt(M)'], 'sigma-p2*S*sqrt(M)')

# %%
cme.lin_reg_rob(OB_UZ_STATS[OB_UZ_STATS['Status']=='after'], ['p1*eta*alpha*sqrt(M)'], 'sigma-p2*S*sqrt(M)')

# %%
cme.regr_plot(PRIOR_OB_UZ_STATS, 'rvx', 'rvxe',\
    'Volatility of Efficient Prices (y) x Estimated Volatility of Efficient Prices (x) : '+CURR)

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS, 'rvx', 'rvxe', True)

# %%
cme.regr_plot(OB_UZ_STATS, 'rvxe', 'ndfpr',\
    'Number of Price Changes (y) x Estimated Volatility of Efficient Prices (x) : '+CURR)

# %%
cme.regr_plot(OB_UZ_STATS, 'rvxe', 'ndfpr',\
    'Number of Price Changes (y) x Estimated Volatility of Efficient Prices (x) : '+CURR, True)

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS, 'rvxe', 'ndfpr', True)

# %%
cme.lin_reg_rob(PRIOR_OB_UZ_STATS, 'rvxe', 'ndfpr', True)

# %%
cme.lin_reg(AFTER_OB_UZ_STATS, 'rvxe', 'ndfpr', True)

# %%
cme.lin_reg_rob(AFTER_OB_UZ_STATS, 'rvxe', 'ndfpr', True)

# %%
cme.regr_plot(OB_UZ_STATS, 'rvxe', 'M',\
    'Number of Trades (y) x Estimated Volatility of Efficient Prices (x) : '+CURR)

# %%
cme.regr_plot(OB_UZ_STATS, 'rvxe', 'M',\
    'Number of Trades (y) x Estimated Volatility of Efficient Prices (x) : '+CURR, True)

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS, 'rvxe', 'M', True)

# %%
cme.lin_reg_rob(PRIOR_OB_UZ_STATS, 'rvxe', 'M', True)

# %%
cme.lin_reg(AFTER_OB_UZ_STATS, 'rvxe', 'M', True)

# %%
cme.lin_reg_rob(AFTER_OB_UZ_STATS, 'rvxe', 'M', True)

# %%
cme.regr_plot(OB_UZ_STATS, 'ndfpr_pred', 'ndfpr',\
    'Realized Number of Price Changes (y) x Predicted Number of Price Changes (x) : '+CURR)

# %%
cme.regr_plot(OB_UZ_STATS, 'ndfpr_pred', 'ndfpr',\
    'Realized Number of Price Changes (y) x Predicted Number of Price Changes (x) : '+CURR, True)

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS, 'ndfpr_pred', 'ndfpr')

# %%
cme.lin_reg_rob(PRIOR_OB_UZ_STATS, 'ndfpr_pred', 'ndfpr')

# %%
cme.lin_reg(AFTER_OB_UZ_STATS, 'ndfpr_pred', 'ndfpr')

# %%
cme.lin_reg_rob(AFTER_OB_UZ_STATS, 'ndfpr_pred', 'ndfpr')

# %%
cme.regr_plot(OB_UZ_STATS, 'ndfpr', 'M',\
    'Number of Trades (y) x Number of Price Changes (x) : '+CURR)

# %%
cme.regr_plot(OB_UZ_STATS, 'ndfpr', 'M',\
    'Number of Trades (y) x Number of Price Changes (x) : '+CURR, True)

# %%
cme.regr_plot(OB_UZ_STATS.loc[:'2015-09-26'], 'ndfpr', 'M',\
    'Number of Trades (y) x Number of Price Changes (x) : '+CURR, True)

# %%
cme.regr_plot(OB_UZ_STATS.loc['2015-09-26':], 'ndfpr', 'M',\
    'Number of Trades (y) x Number of Price Changes (x) : '+CURR, True)

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS.loc[:'2015-09-26'], 'ndfpr', 'M')

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS.loc['2015-09-26':], 'ndfpr', 'M')

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS, 'ndfpr', 'M')

# %%
cme.lin_reg_rob(PRIOR_OB_UZ_STATS, 'ndfpr', 'M')

# %%
cme.lin_reg(AFTER_OB_UZ_STATS, 'ndfpr', 'M')

# %%
cme.lin_reg_rob(AFTER_OB_UZ_STATS, 'ndfpr', 'M')

# %%
cme.regr_plot(PRIOR_OB_UZ_STATS, 'M', 'Volume',\
    'Volume (y) x Number of Trades (x) : '+CURR)

# %%
cme.regr_plot(PRIOR_OB_UZ_STATS, 'M', 'Volume',\
    'Volume (y) x Number of Trades (x) : '+CURR, True)

# %%
cme.lin_reg(PRIOR_OB_UZ_STATS, 'M', 'Volume')

# %%
cme.lin_reg_rob(PRIOR_OB_UZ_STATS, 'M', 'Volume')

# %%
IMBAL_STATS_TS.drop(columns=['eta1']).plot(secondary_y=['Pred_Imbal_Relat'],\
    figsize=(9,6), title='Absolute and relative predictive power of imbalance : EUR');

# %%
TRADE_STATS_TS.plot(secondary_y=['Pred_Trade_Relat'], figsize=(9,6));

# %%
TRADE_STATS_TS.loc[:'2015-09-26'].plot(secondary_y=['Pred_Trade_Relat'], figsize=(9,6));

# %%
TRADE_STATS_TS.loc['2015-09-26':].plot(secondary_y=['Pred_Trade_Relat'], figsize=(9,6));

# %%
OB_UZ_STATS_SPREADS = cme.spread_stats(OB_UZ_STATS)

# %%
cme.time_series_hist_plot(OB_UZ_STATS_SPREADS, 'bid1qty',\
    'Level 1 Bid Average Amount : '+CURR, 0, 80, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS_SPREADS, 'ask1qty',\
    'Level 1 Ask Average Amount : '+CURR, 0, 80, 50)

# %%
OB_UZ_STATS_SPREADS[['bid1qty', 'ask1qty']].plot(figsize=(9,6));

# %%
OB_UZ_STATS_SPREADS[OB_UZ_STATS_SPREADS['Status'] == 'prior'][['bid1qty', 'ask1qty']].mean()/\
    OB_UZ_STATS_SPREADS[OB_UZ_STATS_SPREADS['Status'] == 'after'][['bid1qty', 'ask1qty']].mean()

# %%
OB_UZ_STATS_SPREADS[['bid_adj_qty', 'ask_adj_qty']].plot(figsize=(9,6),\
    title='Adjusted amounts : '+CURR);

# %%
OB_UZ_STATS_SPREADS[['bid_adj_tomid', 'ask_adj_tomid']].plot(figsize=(9,6),\
    title='Adjusted distances between mid and best level(s) expressed in USD : '+CURR);

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='bid_adj_qty', y='bid_adj_tomid',\
                hue='Status', data=OB_UZ_STATS_SPREADS);
plt.title('Adjusted distances between mid and best level(s) expressed in USD (y) vs Adjusted amount (x) : '+CURR);

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='ask_adj_qty', y='ask_adj_tomid',\
                hue='Status', data=OB_UZ_STATS_SPREADS);

# %% [markdown]
# ### Costs

# %%
PRIOR_MEAN_COST = cme.cost_mean(PRIOR_COST_STATS, 100)

# %%
PRIOR_MEAN_COST['Status'] = 'prior'

# %%
AFTER_MEAN_COST = cme.cost_mean(AFTER_COST_STATS, 100)

# %%
AFTER_MEAN_COST['Status'] = 'after'

# %%
MEAN_COST_STATS = pd.concat([PRIOR_MEAN_COST, AFTER_MEAN_COST], sort=False)

# %%
sns.lmplot(x='Trade Qty', y='Avg_Cost', data=PRIOR_MEAN_COST.reset_index(),\
          height=6, aspect=1.5);
plt.title('Average Cost as a function of Trade Amount : '+CURR+' - prior');

# %%
sns.lmplot(x='Trade Qty', y='Avg_Cost', data=PRIOR_MEAN_COST.reset_index(),\
          height=6, aspect=1.5, robust=True);
plt.title('Average Cost as a function of Trade Amount : '+CURR+' - prior');

# %%
cme.lin_reg(cme.cost_mean(PRIOR_COST_STATS, 50).reset_index(), 'Trade Qty', 'Avg_Cost')

# %%
cme.lin_reg_rob(cme.cost_mean(PRIOR_COST_STATS, 50).reset_index(), 'Trade Qty', 'Avg_Cost')

# %%
cme.lin_reg(cme.cost_mean(PRIOR_COST_STATS, 100).reset_index(), 'Trade Qty', 'Avg_Cost')

# %%
cme.lin_reg_rob(cme.cost_mean(PRIOR_COST_STATS, 100).reset_index(), 'Trade Qty', 'Avg_Cost')

# %%
sns.lmplot(x='Trade Qty', y='Avg_Cost', data=AFTER_MEAN_COST.reset_index(),\
          height=6, aspect=1.5);
plt.title('Average Cost as a function of Trade Amount : '+CURR+' - after');

# %%
sns.lmplot(x='Trade Qty', y='Avg_Cost', data=AFTER_MEAN_COST.reset_index(),\
          height=6, aspect=1.5, robust=True);
plt.title('Average Cost as a function of Trade Amount : '+CURR+' - after');

# %%
cme.lin_reg(cme.cost_mean(AFTER_COST_STATS, 50).reset_index(), 'Trade Qty', 'Avg_Cost')

# %%
cme.lin_reg_rob(cme.cost_mean(AFTER_COST_STATS, 50).reset_index(), 'Trade Qty', 'Avg_Cost')

# %%
cme.lin_reg(cme.cost_mean(AFTER_COST_STATS, 100).reset_index(), 'Trade Qty', 'Avg_Cost')

# %%
cme.lin_reg_rob(cme.cost_mean(AFTER_COST_STATS, 100).reset_index(), 'Trade Qty', 'Avg_Cost')

# %%
cme.regr_plot(MEAN_COST_STATS.reset_index(), 'Trade Qty', 'Avg_Cost',\
    'Average Cost as a function of Trade Amount : '+CURR)

# %%
cme.regr_plot(MEAN_COST_STATS.reset_index(), 'Trade Qty', 'Avg_Cost',\
    'Average Cost as a function of Trade Amount : '+CURR, True)

# %% [markdown]
# ## Eta prediction

# %%
cme.plot_eta(TICK_PRIOR, TICK_AFTER,\
    TABLE_MATHIEU.loc['prior']['eta1'], TABLE_MATHIEU.loc['after']['eta1'],\
    TABLE_MATHIEU_ERR.loc['prior']['eta1'], TABLE_MATHIEU_ERR.loc['after']['eta1'],\
    CURR)

# %%
