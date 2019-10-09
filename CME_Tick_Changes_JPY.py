# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
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
CURR = 'JPY'

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
PRIOR_CDATES_LIST = [['6JH5', '20150105'], ['6JH5', '20150106'], ['6JH5', '20150107'],\
    ['6JH5', '20150108'], ['6JH5', '20150109'], ['6JH5', '20150112'], ['6JH5', '20150113'],\
    ['6JH5', '20150114'], ['6JH5', '20150115'], ['6JH5', '20150116'], ['6JH5', '20150119'],\
    ['6JH5', '20150120'], ['6JH5', '20150121'], ['6JH5', '20150122'], ['6JH5', '20150123'],\
    ['6JH5', '20150126'], ['6JH5', '20150127'], ['6JH5', '20150128'], ['6JH5', '20150129'],\
    ['6JH5', '20150130'], ['6JH5', '20150202'], ['6JH5', '20150203'], ['6JH5', '20150204'],\
    ['6JH5', '20150205'], ['6JH5', '20150206'], ['6JH5', '20150209'], ['6JH5', '20150210'],\
    ['6JH5', '20150211'], ['6JH5', '20150212'], ['6JH5', '20150213'], ['6JH5', '20150223'],\
    ['6JH5', '20150224'], ['6JH5', '20150225'], ['6JH5', '20150226'], ['6JH5', '20150227'],\
    ['6JH5', '20150302'], ['6JH5', '20150303'], ['6JH5', '20150304'], ['6JH5', '20150305'],\
    ['6JH5', '20150306'], ['6JH5', '20150309'], ['6JH5', '20150310'], ['6JH5', '20150311'],\
    ['6JH5', '20150312'], ['6JH5', '20150313'], ['6JM5', '20150316'], ['6JM5', '20150317'],\
    ['6JM5', '20150318'], ['6JM5', '20150319'], ['6JM5', '20150320'], ['6JM5', '20150323'],\
    ['6JM5', '20150324'], ['6JM5', '20150325'], ['6JM5', '20150326'], ['6JM5', '20150327'],\
    ['6JM5', '20150330'], ['6JM5', '20150331'], ['6JM5', '20150401'], ['6JM5', '20150402'],\
    ['6JM5', '20150403'], ['6JM5', '20150406'], ['6JM5', '20150407'], ['6JM5', '20150408'],\
    ['6JM5', '20150409'], ['6JM5', '20150410'], ['6JM5', '20150413'], ['6JM5', '20150414'],\
    ['6JM5', '20150415'], ['6JM5', '20150416'], ['6JM5', '20150417'], ['6JM5', '20150420'],\
    ['6JM5', '20150421'], ['6JM5', '20150422'], ['6JM5', '20150423'], ['6JM5', '20150424'],\
    ['6JM5', '20150427'], ['6JM5', '20150428'], ['6JM5', '20150429'], ['6JM5', '20150430'],\
    ['6JM5', '20150501'], ['6JM5', '20150504'], ['6JM5', '20150505'], ['6JM5', '20150506'],\
    ['6JM5', '20150507'], ['6JM5', '20150508'], ['6JM5', '20150511'], ['6JM5', '20150512'],\
    ['6JM5', '20150513'], ['6JM5', '20150514'], ['6JM5', '20150515'], ['6JM5', '20150518'],\
    ['6JM5', '20150519'], ['6JM5', '20150520'], ['6JM5', '20150521'], ['6JM5', '20150522'],\
    ['6JM5', '20150525'], ['6JM5', '20150526'], ['6JM5', '20150527'], ['6JM5', '20150528'],\
    ['6JM5', '20150529'], ['6JM5', '20150601'], ['6JM5', '20150602'], ['6JM5', '20150603'],\
    ['6JM5', '20150604'], ['6JM5', '20150605'], ['6JM5', '20150608'], ['6JM5', '20150609'],\
    ['6JM5', '20150610'], ['6JM5', '20150611'], ['6JU5', '061215'], ['6JU5', '061515'],\
    ['6JU5', '061615'], ['6JU5', '061715'], ['6JU5', '061815'], ['6JU5', '061915']]

# %%
AFTER_CDATES_LIST = [['6JU5', '062215'], ['6JU5', '062315'], ['6JU5', '062415'],\
    ['6JU5', '062515'], ['6JU5', '062615'], ['6JU5', '062915'], ['6JU5', '063015'],\
    ['6JU5', '070115'], ['6JU5', '070215'], ['6JU5', '070315'], ['6JU5', '070615'],\
    ['6JU5', '070715'], ['6JU5', '070815'], ['6JU5', '070915'], ['6JU5', '071015'],\
    ['6JU5', '20150713'], ['6JU5', '20150714'], ['6JU5', '20150715'], ['6JU5', '20150716'],\
    ['6JU5', '20150717'], ['6JU5', '20150720'], ['6JU5', '20150721'], ['6JU5', '20150722'],\
    ['6JU5', '20150723'], ['6JU5', '20150724'], ['6JU5', '20150727'], ['6JU5', '20150728'],\
    ['6JU5', '20150729'], ['6JU5', '20150730'], ['6JU5', '20150731'], ['6JU5', '20150803'],\
    ['6JU5', '20150804'], ['6JU5', '20150805'], ['6JU5', '20150806'], ['6JU5', '20150807'],\
    ['6JU5', '20150810'], ['6JU5', '20150811'], ['6JU5', '20150812'], ['6JU5', '20150813'],\
    ['6JU5', '20150814'], ['6JU5', '20150817'], ['6JU5', '20150818'], ['6JU5', '20150819'],\
    ['6JU5', '20150820'], ['6JU5', '20150821'], ['6JU5', '20150824'], ['6JU5', '20150825'],\
    ['6JU5', '20150826'], ['6JU5', '20150827'], ['6JU5', '20150828'], ['6JU5', '20150831'],\
    ['6JU5', '20150901'], ['6JU5', '20150902'], ['6JU5', '20150903'], ['6JU5', '20150904'],\
    ['6JU5', '20150907'], ['6JU5', '20150908'], ['6JU5', '20150909'], ['6JU5', '20150910'],\
    ['6JU5', '20150911'], ['6JZ5', '20150914'], ['6JZ5', '20150915'], ['6JZ5', '20150916'],\
    ['6JZ5', '20150917'], ['6JZ5', '20150918'], ['6JZ5', '20150921'], ['6JZ5', '20150922'],\
    ['6JZ5', '20150923'], ['6JZ5', '20150924'], ['6JZ5', '20150925'], ['6JZ5', '20150928'],\
    ['6JZ5', '20150929'], ['6JZ5', '20150930'], ['6JZ5', '20151001'], ['6JZ5', '20151002'],\
    ['6JZ5', '20151005'], ['6JZ5', '20151006'], ['6JZ5', '20151007'], ['6JZ5', '20151008'],\
    ['6JZ5', '20151009'], ['6JZ5', '20151012'], ['6JZ5', '20151013'], ['6JZ5', '20151014'],\
    ['6JZ5', '20151015'], ['6JZ5', '20151016'], ['6JZ5', '20151019'], ['6JZ5', '20151020'],\
    ['6JZ5', '20151021'], ['6JZ5', '20151022'], ['6JZ5', '20151023'], ['6JZ5', '20151026'],\
    ['6JZ5', '20151027'], ['6JZ5', '20151028'], ['6JZ5', '20151029'], ['6JZ5', '20151030'],\
    ['6JZ5', '20151102'], ['6JZ5', '20151103'], ['6JZ5', '20151104'], ['6JZ5', '20151105'],\
    ['6JZ5', '20151106'], ['6JZ5', '20151109'], ['6JZ5', '20151110'], ['6JZ5', '20151111'],\
    ['6JZ5', '20151112'], ['6JZ5', '20151113'], ['6JZ5', '20151116'], ['6JZ5', '20151117'],\
    ['6JZ5', '20151118'], ['6JZ5', '20151119'], ['6JZ5', '20151120'], ['6JZ5', '20151123'],\
    ['6JZ5', '20151124'], ['6JZ5', '20151125'], ['6JZ5', '20151126'], ['6JZ5', '20151127'],\
    ['6JZ5', '20151130'], ['6JZ5', '20151201'], ['6JZ5', '20151202'], ['6JZ5', '20151203'],\
    ['6JZ5', '20151204'], ['6JZ5', '20151207'], ['6JZ5', '20151208'], ['6JZ5', '20151209'],\
    ['6JZ5', '20151210'], ['6JZ5', '20151211']]

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
AVG_IMBAL_PRIOR = cme.avg_perc_mat(PRIOR_IMBAL_STATS, pd.to_datetime(PRIOR_CDATES['Date']))
plt.figure(figsize=(9, 6))
sns.heatmap(AVG_IMBAL_PRIOR.iloc[:-1].drop(columns=['Total Cols']),\
    annot=True, fmt=".1f",\
    linewidths=.5, square=True,\
    xticklabels=True,\
    yticklabels=False,\
    cbar=False);

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
    'Depl_Cancel : '+CURR, 0.0, 40000.0, 50)

# %%
cme.time_series_hist_plot(ABSDEPL_STATS_TS, 'Depl_Trades',\
    'Depl_Trades : '+CURR, 0.0, 40000.0, 50)

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
    'DC same-oppo : '+CURR, 0, 40, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Depl_Trade',\
    'Depl_Trade : '+CURR, 0, 40, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DT same-oppo',\
    'DT same-oppo : '+CURR, -1, 22, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DT+F same-oppo',\
    'DT+F same-oppo : '+CURR, -1, 9, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Fill same-oppo',\
    'Filled : Same - Opposite : '+CURR, -15, 35, 50)

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
    'Spread in Ticks : '+CURR, 0.8, 2.5, 50)

# %%
cme.twspr_plot_USD(OB_UZ_STATS, CURR)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'eta1',\
    '$\eta$ : '+CURR, 0, 0.65, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'chgavg',\
    'Average Price Change : '+CURR, 0.4, 1.4, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'rvxe',\
    'Estimated Volatility of Efficient Prices : '+CURR, 0, 0.015, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'ndfpr',\
    'Number of Price Changes : '+CURR, 0, 15000, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'M',\
    'Number of Trades : '+CURR, 0, 70000, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'Volume',\
    'Volume : '+CURR, 0, 250000, 50)

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
OB_UZ_STATS_SPREADS = cme.spread_stats(OB_UZ_STATS)

# %%
cme.time_series_hist_plot(OB_UZ_STATS_SPREADS, 'bid1qty',\
    'Level 1 Bid Average Amount : '+CURR, 0, 130, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS_SPREADS, 'ask1qty',\
    'Level 1 Ask Average Amount : '+CURR, 0, 130, 50)

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
