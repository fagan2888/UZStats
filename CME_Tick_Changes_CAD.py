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
CURR = 'CAD'

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
PRIOR_CDATES_LIST = [['6CH6', '010416'], ['6CH6', '010516'], ['6CH6', '010616'],\
    ['6CH6', '010716'], ['6CH6', '010816'], ['6CH6', '011116'], ['6CH6', '011216'],\
    ['6CH6', '011316'], ['6CH6', '011416'], ['6CH6', '011516'], ['6CH6', '011816'],\
    ['6CH6', '011916'], ['6CH6', '012016'], ['6CH6', '012116'], ['6CH6', '012216'],\
    ['6CH6', '012516'], ['6CH6', '012616'], ['6CH6', '012716'], ['6CH6', '012816'],\
    ['6CH6', '012916'], ['6CH6', '20160201'], ['6CH6', '20160202'], ['6CH6', '20160203'],\
    ['6CH6', '20160204'], ['6CH6', '20160205'], ['6CH6', '20160208'], ['6CH6', '20160209'],\
    ['6CH6', '20160210'], ['6CH6', '20160211'], ['6CH6', '20160212'], ['6CH6', '20160215'],\
    ['6CH6', '20160216'], ['6CH6', '20160217'], ['6CH6', '20160218'], ['6CH6', '20160219'],\
    ['6CH6', '20160222'], ['6CH6', '20160223'], ['6CH6', '20160224'], ['6CH6', '20160225'],\
    ['6CH6', '20160226'], ['6CH6', '20160229'], ['6CH6', '20160301'], ['6CH6', '20160302'],\
    ['6CH6', '20160303'], ['6CH6', '20160304'], ['6CH6', '20160307'], ['6CH6', '20160308'],\
    ['6CH6', '20160309'], ['6CH6', '20160310'], ['6CH6', '20160311'], ['6CM6', '20160314'],\
    ['6CM6', '20160315'], ['6CM6', '20160316'], ['6CM6', '20160317'], ['6CM6', '20160318'],\
    ['6CM6', '20160321'], ['6CM6', '20160322'], ['6CM6', '20160323'], ['6CM6', '20160324'],\
    ['6CM6', '20160328'], ['6CM6', '20160329'], ['6CM6', '20160330'], ['6CM6', '20160331'],\
    ['6CM6', '20160401'], ['6CM6', '20160404'], ['6CM6', '20160405'], ['6CM6', '20160406'],\
    ['6CM6', '20160407'], ['6CM6', '20160408'], ['6CM6', '20160411'], ['6CM6', '20160412'],\
    ['6CM6', '20160413'], ['6CM6', '20160414'], ['6CM6', '20160415'], ['6CM6', '20160418'],\
    ['6CM6', '20160419'], ['6CM6', '20160420'], ['6CM6', '20160421'], ['6CM6', '20160422'],\
    ['6CM6', '20160425'], ['6CM6', '20160426'], ['6CM6', '20160427'], ['6CM6', '20160428'],\
    ['6CM6', '20160429'], ['6CM6', '20160502'], ['6CM6', '20160503'], ['6CM6', '20160504'],\
    ['6CM6', '20160505'], ['6CM6', '20160506'], ['6CM6', '20160509'], ['6CM6', '20160510'],\
    ['6CM6', '20160511'], ['6CM6', '20160512'], ['6CM6', '20160513'], ['6CM6', '20160516'],\
    ['6CM6', '20160517'], ['6CM6', '20160518'], ['6CM6', '20160519'], ['6CM6', '20160520'],\
    ['6CM6', '20160523'], ['6CM6', '20160524'], ['6CM6', '20160525'], ['6CM6', '20160526'],\
    ['6CM6', '20160527'], ['6CM6', '20160530'], ['6CM6', '20160531'], ['6CM6', '20160601'],\
    ['6CM6', '20160602'], ['6CM6', '20160603'], ['6CM6', '20160606'], ['6CM6', '20160607'],\
    ['6CM6', '20160608'], ['6CM6', '20160609'], ['6CU6', '061016'], ['6CU6', '061316'],\
    ['6CU6', '061416'], ['6CU6', '061516'], ['6CU6', '061616'], ['6CU6', '061716'],\
    ['6CU6', '062016'], ['6CU6', '062116'], ['6CU6', '062216'], ['6CU6', '062316'],\
    ['6CU6', '062416'], ['6CU6', '062716'], ['6CU6', '062816'], ['6CU6', '062916'],\
    ['6CU6', '063016'], ['6CU6', '070116'], ['6CU6', '070416'], ['6CU6', '070516'],\
    ['6CU6', '070616'], ['6CU6', '070716'], ['6CU6', '070816']]

# %%
AFTER_CDATES_LIST = [['6CU6', '071116'], ['6CU6', '071216'], ['6CU6', '071316'],\
    ['6CU6', '071416'], ['6CU6', '071516'], ['6CU6', '071816'], ['6CU6', '071916'],\
    ['6CU6', '072016'], ['6CU6', '072116'], ['6CU6', '072216'], ['6CU6', '072516'],\
    ['6CU6', '072616'], ['6CU6', '072716'], ['6CU6', '072816'], ['6CU6', '072916'],\
    ['6CU6', '080116'], ['6CU6', '080216'], ['6CU6', '080316'], ['6CU6', '080416'],\
    ['6CU6', '080516'], ['6CU6', '080816'], ['6CU6', '080916'], ['6CU6', '081016'],\
    ['6CU6', '081116'], ['6CU6', '081216'], ['6CU6', '081516'], ['6CU6', '081616'],\
    ['6CU6', '081716'], ['6CU6', '081816'], ['6CU6', '081916'], ['6CU6', '082216'],\
    ['6CU6', '082316'], ['6CU6', '082416'], ['6CU6', '082516'], ['6CU6', '082616'],\
    ['6CU6', '082916'], ['6CU6', '083016'], ['6CU6', '083116'], ['6CU6', '090116'],\
    ['6CU6', '090216'], ['6CU6', '090516'], ['6CU6', '090616'], ['6CU6', '090716'],\
    ['6CU6', '090816'], ['6CU6', '090916'], ['6CU6', '091216'], ['6CU6', '091316'],\
    ['6CU6', '091416'], ['6CU6', '091516'], ['6CZ6', '20160916'], ['6CZ6', '20160919'],\
    ['6CZ6', '20160920'], ['6CZ6', '20160921'], ['6CZ6', '20160922'], ['6CZ6', '20160923'],\
    ['6CZ6', '20160926'], ['6CZ6', '20160927'], ['6CZ6', '20160928'], ['6CZ6', '20160929'],\
    ['6CZ6', '20160930'], ['6CZ6', '20161003'], ['6CZ6', '20161004'], ['6CZ6', '20161005'],\
    ['6CZ6', '20161006'], ['6CZ6', '20161007'], ['6CZ6', '20161010'], ['6CZ6', '20161011'],\
    ['6CZ6', '20161012'], ['6CZ6', '20161013'], ['6CZ6', '20161014'], ['6CZ6', '20161017'],\
    ['6CZ6', '20161018'], ['6CZ6', '20161019'], ['6CZ6', '20161020'], ['6CZ6', '20161021'],\
    ['6CZ6', '20161024'], ['6CZ6', '20161025'], ['6CZ6', '20161026'], ['6CZ6', '20161027'],\
    ['6CZ6', '20161028'], ['6CZ6', '20161031'], ['6CZ6', '20161101'], ['6CZ6', '20161102'],\
    ['6CZ6', '20161103'], ['6CZ6', '20161104'], ['6CZ6', '20161107'], ['6CZ6', '20161108'],\
    ['6CZ6', '20161109'], ['6CZ6', '20161110'], ['6CZ6', '20161111'], ['6CZ6', '20161114'],\
    ['6CZ6', '20161115'], ['6CZ6', '20161116'], ['6CZ6', '20161117'], ['6CZ6', '20161118'],\
    ['6CZ6', '20161121'], ['6CZ6', '20161122'], ['6CZ6', '20161123'], ['6CZ6', '20161124'],\
    ['6CZ6', '20161125'], ['6CZ6', '20161128'], ['6CZ6', '20161129'], ['6CZ6', '20161130'],\
    ['6CZ6', '20161201'], ['6CZ6', '20161202'], ['6CZ6', '20161205'], ['6CZ6', '20161206'],\
    ['6CZ6', '20161207'], ['6CZ6', '20161208'], ['6CZ6', '20161209'], ['6CZ6', '20161212'],\
    ['6CZ6', '20161213'], ['6CZ6', '20161214'], ['6CZ6', '20161215'], ['6CZ6', '20161216'],\
    ['w6CH7', '20161219'], ['w6CH7', '20161220'], ['w6CH7', '20161221'], ['w6CH7', '20161222'],\
    ['w6CH7', '20161223'], ['w6CH7', '20161227'], ['w6CH7', '20161228'], ['w6CH7', '20161229'],\
    ['w6CH7', '20161230'], ['x6CH7', '010317'], ['x6CH7', '010417'], ['x6CH7', '010517'],\
    ['x6CH7', '010617'], ['x6CH7', '010917'], ['x6CH7', '011017'], ['x6CH7', '011117'],\
    ['x6CH7', '011217'], ['x6CH7', '011317'], ['x6CH7', '011617'], ['x6CH7', '011717'],\
    ['x6CH7', '011817'], ['x6CH7', '011917'], ['x6CH7', '012017'], ['x6CH7', '012317'],\
    ['x6CH7', '012417'], ['x6CH7', '012517'], ['x6CH7', '012617'], ['x6CH7', '012717'],\
    ['x6CH7', '013017']]

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
    'DC same-oppo : '+CURR, 0, 35, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Depl_Trade',\
    'Depl_Trade : '+CURR, 0, 40, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DT same-oppo',\
    'DT same-oppo : '+CURR, -2, 20, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DT+F same-oppo',\
    'DT+F same-oppo : '+CURR, -1, 3, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Fill same-oppo',\
    'Filled : Same - Opposite : '+CURR, -5, 30, 50)

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
    'Spread in Ticks : '+CURR, 1, 6, 50)

# %%
cme.twspr_plot_USD(OB_UZ_STATS, CURR)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'eta1',\
    '$\eta$ : '+CURR, 0, 0.55, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'chgavg',\
    'Average Price Change : '+CURR, 0.4, 1.4, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'rvxe',\
    'Estimated Volatility of Efficient Prices : '+CURR, 0, 0.015, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'ndfpr',\
    'Number of Price Changes : '+CURR, 0, 12000, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'M',\
    'Number of Trades : '+CURR, 0, 40000, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'Volume',\
    'Volume : '+CURR, 0, 120000, 50)

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
    'Level 1 Bid Average Amount : '+CURR, 0, 60, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS_SPREADS, 'ask1qty',\
    'Level 1 Ask Average Amount : '+CURR, 0, 60, 50)

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
