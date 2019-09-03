# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The Robert and Rosenbaum Uncertainty Zones model

# %% [markdown]
# # An application to FX Futures in Brazil

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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# %%
from pandas.plotting import scatter_matrix

# %%
pd.set_option('display.max_columns', 50)

# %%
pd.set_option('display.max_rows', 100)

# %% [markdown]
# ## File paths

# %%
PATHPROJ = '/Users/marcoscscarreira/Documents/X/CME project/'
PATHIN = PATHPROJ+'data/'
PATHOUT = PATHPROJ+'dfs/'

# %% [markdown]
# ## Inputs

# %% [markdown]
# #### Parameters

# %%
TS = 0.5
START_TIME = pd.to_timedelta('09:00:00')
END_TIME = pd.to_timedelta('18:15:00')
TRADING_HOURS = 9.25

# %% [markdown]
# #### File lists

# %%
FILESDOL2017 = [
    'DOLG1720170103.csv', 'DOLG1720170104.csv', 'DOLG1720170105.csv', \
    'DOLG1720170106.csv', 'DOLG1720170109.csv', 'DOLG1720170110.csv', \
    'DOLG1720170111.csv', 'DOLG1720170112.csv', 'DOLG1720170113.csv', \
    'DOLG1720170116.csv', 'DOLG1720170117.csv', 'DOLG1720170118.csv', \
    'DOLG1720170119.csv', 'DOLG1720170120.csv', 'DOLG1720170123.csv', \
    'DOLG1720170124.csv', 'DOLG1720170126.csv', 'DOLG1720170127.csv', \
    'DOLG1720170130.csv', 'DOLH1720170131.csv', \
    'DOLH1720170201.csv', 'DOLH1720170202.csv', 'DOLH1720170203.csv', \
    'DOLH1720170207.csv', 'DOLH1720170208.csv', \
    'DOLH1720170209.csv', 'DOLH1720170210.csv', 'DOLH1720170213.csv', \
    'DOLH1720170214.csv', 'DOLH1720170215.csv', 'DOLH1720170216.csv', \
    'DOLH1720170217.csv', 'DOLH1720170220.txt', 'DOLH1720170221.txt', \
    'DOLH1720170222.txt', 'DOLH1720170223.txt']

# %%
DATES_DOL = [pd.to_datetime(f[6:14], format='%Y%m%d') for f in FILESDOL2017]
PRODUCT_DOL = [f[:3] for f in FILESDOL2017]
CONTRACT_DOL = [f[3:6] for f in FILESDOL2017]

# %%
FILESWDO2017 = [
    'WDOG1720170103.csv', 'WDOG1720170104.csv', 'WDOG1720170105.csv', \
    'WDOG1720170106.csv', 'WDOG1720170109.csv', 'WDOG1720170110.csv', \
    'WDOG1720170111.csv', 'WDOG1720170112.csv', 'WDOG1720170113.csv', \
    'WDOG1720170116.csv', 'WDOG1720170117.csv', 'WDOG1720170118.csv', \
    'WDOG1720170119.csv', 'WDOG1720170120.csv', 'WDOG1720170123.csv', \
    'WDOG1720170124.csv', 'WDOG1720170126.csv', 'WDOG1720170127.csv', \
    'WDOG1720170130.csv', 'WDOH1720170131.csv', \
    'WDOH1720170201.csv', 'WDOH1720170202.csv', 'WDOH1720170203.csv', \
    'WDOH1720170207.csv', 'WDOH1720170208.csv', \
    'WDOH1720170209.csv', 'WDOH1720170210.csv', 'WDOH1720170213.csv', \
    'WDOH1720170214.csv', 'WDOH1720170215.csv', 'WDOH1720170216.csv', \
    'WDOH1720170217.csv', 'WDOH1720170220.txt', 'WDOH1720170221.txt', \
    'WDOH1720170222.txt', 'WDOH1720170223.txt']

# %%
DATES_WDO = [pd.to_datetime(f[6:14], format='%Y%m%d') for f in FILESWDO2017]
PRODUCT_WDO = [f[:3] for f in FILESWDO2017]
CONTRACT_WDO = [f[3:6] for f in FILESWDO2017]

# %% [markdown]
# ## Uncertainty Zones and Top of Book statistics

# %%
DF_STATS = pd.DataFrame()
for j in range(len(FILESDOL2017)):
    new_row_1 = pd.read_csv(PATHOUT+FILESDOL2017[j][:-4]+'_OBstats.csv', index_col=0)
    new_row_2 = pd.read_csv(PATHOUT+FILESDOL2017[j][:-4]+'_UZstats.csv', index_col=0)
    new_row = pd.concat([new_row_1, new_row_2], axis=1, sort=False)
    new_row['lambda1'] = pd.read_csv(PATHOUT+FILESDOL2017[j][:-4]+'_CAticks.csv', index_col=0)\
        .set_index('Li').loc[1]['lamb']
    DF_STATS = DF_STATS.append(new_row)
for j in range(len(FILESWDO2017)):
    new_row_1 = pd.read_csv(PATHOUT+FILESWDO2017[j][:-4]+'_OBstats.csv', index_col=0)
    new_row_2 = pd.read_csv(PATHOUT+FILESWDO2017[j][:-4]+'_UZstats.csv', index_col=0)
    new_row = pd.concat([new_row_1, new_row_2], axis=1, sort=False)
    new_row['lambda1'] = pd.read_csv(PATHOUT+FILESWDO2017[j][:-4]+'_CAticks.csv', index_col=0)\
        .set_index('Li').loc[1]['lamb']
    DF_STATS = DF_STATS.append(new_row)
DF_STATS['Dates'] = DATES_DOL+DATES_WDO
DF_STATS['Product'] = PRODUCT_DOL+PRODUCT_WDO
DF_STATS['Contract'] = CONTRACT_DOL+CONTRACT_WDO
DF_STATS.set_index(['Product', 'Contract', 'Dates'], inplace=True)
DF_STATS.sort_index(inplace=True)
DF_STATS['ndfpr_pred'] = TRADING_HOURS*3600/DF_STATS['duration']
#DF_STATS['ndfpr_pred'] = ((DF_STATS['rvx']*DF_STATS['spot_avg']/TS)**2)/(2*DF_STATS['eta1'])
DF_STATS['eta*alpha*sqrt(M)'] = DF_STATS['eta1']*TS*np.sqrt(DF_STATS['M'])
DF_STATS['S*sqrt(M)'] = DF_STATS['twspr1']*TS*np.sqrt(DF_STATS['M'])
DF_STATS['sigma'] = DF_STATS['rvx']*DF_STATS['spot_avg']
DF_STATS['p1*eta*alpha*sqrt(M)'] = np.where(DF_STATS.reset_index()['Product']=='DOL', 1.2119, 1.9221)*DF_STATS['eta*alpha*sqrt(M)']
DF_STATS['sigma-p2*S*sqrt(M)'] = DF_STATS['sigma']-\
    np.where(DF_STATS.reset_index()['Product']=='DOL', 0.0870, 0.0364)*DF_STATS['S*sqrt(M)']

# %%
DF_STATS_PLOT = DF_STATS.copy().reset_index()

# %%
#DF_STATS

# %% [markdown]
# ## Tables

# %% [markdown]
# ### Table 1

# %%
DF_STATS[['chgavg', 'ndfpr_pred', 'ndfpr', 'M', 'eta1',\
    'S1', 'lambda1', 'twspr1', 'duration', 'dt_avg', 'rvxe', 'spot_avg']]\
    .groupby('Product').mean()

# %%
DF_STATS.loc['DOL']['lambda1'].hist();


# %% [markdown]
# ## Charts

# %%
def lin_reg(data, independent, dependent, logdata=False):
    X = data[independent]
    Y = data[dependent]
    if logdata:
        X = np.log(X)
        Y = np.log(Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X) 
    print_model = model.summary()
    print(print_model)


# %%
def lin_reg_sa(x, y, logdata=False):
    X = x
    Y = y
    if logdata:
        X = np.log(X)
        Y = np.log(Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X) 
    print_model = model.summary()
    print(print_model)


# %%
def log_df(data, dependent, independent, hue):
    return pd.concat([np.log(data[dependent]), np.log(data[independent]), data[hue]], axis=1)


# %%
sns.lmplot(x='eta*alpha*sqrt(M)', y='sigma', hue='Product', data=DF_STATS_PLOT,\
          height=6, aspect=1.5);
plt.title(r' Cloud ('
          r'$\eta\alpha\sqrt{M}, \sigma)$');
plt.xlabel(r'$\eta\alpha\sqrt{M}$');
plt.ylabel(r'$\sigma$');

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['eta*alpha*sqrt(M)', 'S*sqrt(M)'], 'sigma')

# %%
sns.lmplot(x='p1*eta*alpha*sqrt(M)', y='sigma-p2*S*sqrt(M)', hue='Product', data=DF_STATS_PLOT,\
          height=6, aspect=1.5);
plt.title(r' Cloud ('
          r'$p_{1}\eta\alpha\sqrt{M}, \sigma-p_{2}S\sqrt{M})$');
plt.xlabel(r'$p_{1}\eta\alpha\sqrt{M}$');
plt.ylabel(r'$\sigma-p_{2}S\sqrt{M}$');

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['p1*eta*alpha*sqrt(M)'], 'sigma-p2*S*sqrt(M)')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['p1*eta*alpha*sqrt(M)'], 'sigma-p2*S*sqrt(M)')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['eta*alpha*sqrt(M)'], 'sigma')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['eta*alpha*sqrt(M)'], 'sigma')

# %%
sns.lmplot(x='ndfpr', y='M', hue='Product', aspect=1, \
            height=8, data=DF_STATS_PLOT);

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], 'ndfpr', 'M')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], 'ndfpr', 'M')

# %%
scatter_matrix(DF_STATS_PLOT[['rvx','eta1','ndfpr','M']], figsize=(8, 8));

# %%
sns.lmplot(x='ndfpr', y='M', aspect=1, \
            height=4, data=DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL']);

# %%
sns.lmplot(x='ndfpr', y='M', aspect=1, \
            height=4, data=DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO']);

# %%
sns.lmplot(x='ndfpr_pred', y='ndfpr', hue='Product', aspect=1, \
            height=8, data=DF_STATS_PLOT);
plt.title('Number of price changes realized (y) and predicted (x)');

# %%
sns.lmplot(x='ndfpr_pred', y='M', hue='Product', aspect=1, \
            height=8, data=DF_STATS_PLOT);
plt.title('Number of trades (y) versus price changes predicted (x)');

# %%
sns.lmplot(x='ndfpr', y='M', hue='Product', aspect=1, \
            height=8, data=DF_STATS_PLOT);
plt.title('Number of trades (y) versus price changes (x)');

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['ndfpr_pred'], 'ndfpr')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['ndfpr_pred'], 'ndfpr')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['ndfpr'], 'M')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['ndfpr'], 'M')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['ndfpr_pred'], 'M')

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['ndfpr_pred'], 'M')

# %%
sns.lmplot(x='rvx', y='M', hue='Product', aspect=1, \
            height=8, data=DF_STATS_PLOT);

# %%
sns.lmplot(x='rvx', y='M', hue='Product', aspect=1, \
            height=8, data=log_df(DF_STATS_PLOT, 'rvx', 'M', 'Product'));

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['rvx'], 'ndfpr', True)

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['rvx'], 'M', True)

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['rvx'], 'ndfpr', True)

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['rvx'], 'M', True)

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['eta1', 'rvx'], 'M', True)

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['eta1', 'rvx'], 'M', True)

# %%
sns.lmplot(x='rvx', y='rvxe', hue='Product', aspect=1, \
            height=8, data=DF_STATS_PLOT);

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL'], ['rvx'], 'rvxe', True)

# %%
lin_reg(DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO'], ['rvx'], 'rvxe', True)

# %%
ax = DF_STATS.plot.scatter(x='duration', y='dt_avg', figsize=(9,9));
ax.set_aspect('equal')

# %%
plt.figure(figsize=(9,9))
sns.scatterplot(x='chgavg', y='eta1', hue='Product',\
            data=DF_STATS_PLOT);

# %%
plt.figure(figsize=(9,9))
sns.scatterplot(x='twspr1', y='eta1', hue='Product',\
            data=DF_STATS_PLOT);

# %%
plt.figure(figsize=(9,9))
sns.scatterplot(x='rvxe', y='eta1', hue='Product',\
            data=DF_STATS_PLOT);

# %%
sns.lmplot(x='rvx', y='rvp', hue='Product', aspect=1, \
            height=8, data=DF_STATS_PLOT);

# %%
plt.figure(figsize=(9,9))
sns.lineplot(x='Dates', y='rvp', hue='Product', data=DF_STATS_PLOT);
plt.xticks(rotation=45);
plt.title('Volatility of the traded prices');

# %%
plt.figure(figsize=(9,9))
sns.lineplot(x='Dates', y='rvx', hue='Product', data=DF_STATS_PLOT);
plt.xticks(rotation=45);
plt.title('Volatility of the efficient prices');

# %%
plt.figure(figsize=(9,9))
sns.lineplot(x='Dates', y='rvp', hue='Product', data=DF_STATS_PLOT);
plt.xticks(rotation=45);

# %%
plt.figure(figsize=(9,9))
sns.lineplot(x='Dates', y='twspr1', hue='Product', data=DF_STATS_PLOT);
plt.xticks(rotation=45);

# %% [markdown]
# ## States - Orders and Trades

# %%
DF_STATS_IMBAL = pd.DataFrame()
for j in range(len(FILESDOL2017)):
    new_row = pd.read_csv(PATHOUT+FILESDOL2017[j][:-4]+'_OTtrans.csv', index_col=0)
    new_row['Dates'] = DATES_DOL[j].strftime('%Y-%m-%d')
    new_row['Product'] = FILESDOL2017[j][:3]
    new_row['Contract'] = FILESDOL2017[j][3:6]
    DF_STATS_IMBAL = DF_STATS_IMBAL.append(new_row)
for j in range(len(FILESWDO2017)):
    new_row = pd.read_csv(PATHOUT+FILESWDO2017[j][:-4]+'_OTtrans.csv', index_col=0)
    new_row['Dates'] = DATES_WDO[j].strftime('%Y-%m-%d')
    new_row['Product'] = FILESWDO2017[j][:3]
    new_row['Contract'] = FILESWDO2017[j][3:6]
    DF_STATS_IMBAL = DF_STATS_IMBAL.append(new_row)
DF_STATS_IMBAL.reset_index()
DF_STATS_IMBAL.set_index(['Product', 'Contract', 'Dates'], inplace=True)
DF_STATS_IMBAL.sort_index(inplace=True)


# %%
def get_imbal_mat(product, date):
    mat_values = DF_STATS_IMBAL.loc[product, :, date].values
    mat_df = pd.DataFrame(mat_values,\
        columns=['Trade_Bid', 'Imbal_Bid',\
            'Neutral', 'Imbal_Ask', 'Trade_Ask'],\
        index=['Trade_Bid', 'Imbal_Bid',\
            'Neutral', 'Imbal_Ask', 'Trade_Ask'])
    return mat_df


# %%
def perc_mat(mat):
    sum_mat = mat.to_numpy().sum()
    norm_mat = (100*mat/sum_mat)
    norm_mat['Total Cols'] = norm_mat.sum(axis=1)
    norm_mat.loc['Total Rows'] = norm_mat.sum(axis=0)
    return norm_mat


# %%
def perc_mat_r(mat, dec=2):
    sum_mat = mat.to_numpy().sum()
    norm_mat = (100*mat/sum_mat)
    norm_mat['Total Cols'] = norm_mat.sum(axis=1)
    norm_mat.loc['Total Rows'] = norm_mat.sum(axis=0)
    return norm_mat.round(dec)


# %%
IMBAL_DATES = DF_STATS_IMBAL.index.get_level_values('Dates').unique().values


# %%
def avg_perc_mat(product, dates):
    date_count = 1
    result = perc_mat(get_imbal_mat(product, dates[date_count-1]))
    for date in dates[1:]:
        date_count += 1
        result += (perc_mat(get_imbal_mat(product, dates[date_count-1]))-result)/date_count
    return result


# %%
def summ_imbal(mat):
    pred_ib = (mat.loc['Imbal_Bid']['Trade_Bid']-mat.loc['Imbal_Bid']['Trade_Ask'])
    pred_ia = (mat.loc['Imbal_Ask']['Trade_Ask']-mat.loc['Imbal_Ask']['Trade_Bid'])
    pred_imb = (pred_ib+pred_ia)/2
    pred_ibr = pred_ib/mat.loc['Imbal_Bid']['Total Cols']*100
    pred_iar = pred_ia/mat.loc['Imbal_Ask']['Total Cols']*100
    pred_imbr = (pred_ibr+pred_iar)/2
    return [pred_imb, pred_imbr]


# %%
def time_series_imbal(product, dates):
    list_imbal = []
    for date in dates:
        new_mat = perc_mat(get_imbal_mat(product, date))
        new_row = summ_imbal(new_mat)
        list_imbal = list_imbal+[new_row]
    df_imbal = pd.DataFrame(list_imbal, columns=['Pred_Imbal', 'Pred_Imbal_Relat'],\
            index=dates)
    return df_imbal


# %%
def summ_trade(mat):
    pred_tb = (mat.loc['Trade_Bid']['Trade_Bid']-mat.loc['Trade_Bid']['Trade_Ask'])
    pred_ta = (mat.loc['Trade_Ask']['Trade_Ask']-mat.loc['Trade_Ask']['Trade_Bid'])
    pred_trd = (pred_tb+pred_ta)/2
    pred_tbr = pred_tb/mat.loc['Trade_Bid']['Total Cols']*100
    pred_tar = pred_ta/mat.loc['Trade_Ask']['Total Cols']*100
    pred_trdr = (pred_tbr+pred_tar)/2
    return [pred_trd, pred_trdr]


# %%
def time_series_imbal_trd(product, dates):
    list_imbal = []
    for date in dates:
        new_mat = perc_mat(get_imbal_mat(product, date))
        new_row = summ_trade(new_mat)
        list_imbal = list_imbal+[new_row]
    df_imbal = pd.DataFrame(list_imbal, columns=['Pred_Trade', 'Pred_Trade_Relat'],\
            index=dates)
    return df_imbal


# %%
DOL_ETA = DF_STATS_PLOT[DF_STATS_PLOT['Product']=='DOL']['eta1'].values

# %%
DOL_IMBAL_TS = pd.concat([time_series_imbal('DOL', IMBAL_DATES),\
                         time_series_imbal_trd('DOL', IMBAL_DATES)], axis=1, sort=False)

# %%
DOL_IMBAL_TS['eta1'] = DOL_ETA
DOL_IMBAL_TS['Product'] = 'DOL'
DOL_IMBAL_TS.index.names=['Dates']

# %%
DOL_IMBAL_TS[['Pred_Imbal', 'Pred_Imbal_Relat']].plot(secondary_y=['Pred_Imbal_Relat'],\
    figsize=(9,6), title='Absolute and relative predictive power of imbalance for the DOL');

# %%
DOL_IMBAL_TS[['Pred_Trade', 'Pred_Trade_Relat']].plot(secondary_y=['Pred_Trade_Relat'], figsize=(9,6));

# %%
DOL_IMBAL_TS[['eta1', 'Pred_Imbal_Relat']].plot(secondary_y=['Pred_Imbal_Relat'], figsize=(9,6));

# %%
sns.lmplot(x='eta1', y='Pred_Imbal_Relat', data=DOL_IMBAL_TS,\
           aspect=1, height=6);
plt.title('Relative predictive power of imbalance and $\eta$ for the DOL');

# %%
WDO_ETA = DF_STATS_PLOT[DF_STATS_PLOT['Product']=='WDO']['eta1'].values

# %%
WDO_IMBAL_TS = pd.concat([time_series_imbal('WDO', IMBAL_DATES),\
                         time_series_imbal_trd('WDO', IMBAL_DATES)], axis=1, sort=False)

# %%
WDO_IMBAL_TS['eta1'] = WDO_ETA
WDO_IMBAL_TS['Product'] = 'WDO'
WDO_IMBAL_TS.index.names=['Dates']

# %%
WDO_IMBAL_TS[['Pred_Imbal', 'Pred_Imbal_Relat']].plot(secondary_y=['Pred_Imbal_Relat'],\
    figsize=(9,6), title='Absolute and relative predictive power of imbalance for the WDO');

# %%
WDO_IMBAL_TS[['Pred_Trade', 'Pred_Trade_Relat']].plot(secondary_y=['Pred_Trade_Relat'], figsize=(9,6));

# %%
WDO_IMBAL_TS[['eta1', 'Pred_Imbal_Relat']].plot(secondary_y=['Pred_Imbal_Relat'], figsize=(9,6));

# %%
sns.lmplot(x='eta1', y='Pred_Imbal_Relat', data=WDO_IMBAL_TS,\
           aspect=1, height=6);
plt.title('Relative predictive power of imbalance and $\eta$ for the WDO');

# %%
ALL_IMBAL_TS = pd.concat([WDO_IMBAL_TS.reset_index(), DOL_IMBAL_TS.reset_index()], sort=False)

# %%
plt.figure(figsize=(8, 8))
sns.scatterplot(x='eta1', y='Pred_Imbal_Relat', hue='Product',\
           data=ALL_IMBAL_TS);
plt.title('Relative predictive power of imbalance and $\eta$ for DOL and WDO');

# %%
get_imbal_mat('DOL','2017-01-03')

# %%
get_imbal_mat('WDO','2017-01-03')

# %%
perc_mat(get_imbal_mat('DOL','2017-01-03'))

# %%
perc_mat(get_imbal_mat('WDO','2017-01-03'))

# %%
perc_mat(get_imbal_mat('DOL','2017-01-04'))

# %%
perc_mat(get_imbal_mat('WDO','2017-01-04'))

# %%
np.round(avg_perc_mat('DOL', IMBAL_DATES),2)

# %%
np.round(avg_perc_mat('WDO', IMBAL_DATES),2)

# %% [markdown]
# ## States - Depletions and Fills

# %%
DF_STATS_DEPL = pd.DataFrame()
for j in range(len(FILESDOL2017)):
    new_row = pd.read_csv(PATHOUT+FILESDOL2017[j][:-4]+'_RDFtrans.csv',\
                         index_col=0, header=[0,1])
    new_row['Dates'] = DATES_DOL[j].strftime('%Y-%m-%d')
    new_row['Product'] = FILESDOL2017[j][:3]
    new_row['Contract'] = FILESDOL2017[j][3:6]
    DF_STATS_DEPL = DF_STATS_DEPL.append(new_row)
for j in range(len(FILESWDO2017)):
    new_row = pd.read_csv(PATHOUT+FILESWDO2017[j][:-4]+'_RDFtrans.csv',\
                         index_col=0, header=[0,1])
    new_row['Dates'] = DATES_WDO[j].strftime('%Y-%m-%d')
    new_row['Product'] = FILESWDO2017[j][:3]
    new_row['Contract'] = FILESWDO2017[j][3:6]
    DF_STATS_DEPL = DF_STATS_DEPL.append(new_row)
DF_STATS_DEPL.reset_index()
DF_STATS_DEPL.set_index(['Product', 'Contract', 'Dates'], inplace=True)
DF_STATS_DEPL.sort_index(inplace=True)


# %%
def get_depl_mat(product, date):
    mat_values = DF_STATS_DEPL.loc[product, :, date].values
    mat_cols = pd.MultiIndex.from_tuples(\
        [('same', ' D C '),\
        ('same', ' D T '),\
        ('same', 'D T+F'),\
        ('same', '  F  '),\
        ('oppo', ' D C '),\
        ('oppo', ' D T '),\
        ('oppo', 'D T+F'),\
        ('oppo', '  F  ')])
    mat_df = pd.DataFrame(mat_values,\
        columns=mat_cols,\
        index=[' D C ', ' D T ',\
            'D T+F', '  F  '])
    return mat_df


# %%
def summ_depl(mat):
    depl_c = mat.loc[' D C ']['Total Cols'].sum()
    depl_t = mat.loc[' D T ']['Total Cols'].sum()
    depl_tf = mat.loc['D T+F']['Total Cols'].sum()
    depl_c_s = mat.loc[' D C ']['same', '  F  ']-mat.loc[' D C ']['oppo', '  F  ']
    depl_t_s = mat.loc[' D T ']['same', '  F  ']-mat.loc[' D T ']['oppo', '  F  ']
    depl_tf_s = mat.loc['D T+F']['same', '  F  ']-mat.loc['D T+F']['oppo', '  F  ']
    depl_f_s = mat.loc['  F  ']['same'].sum()-mat.loc['  F  ']['oppo'].sum()
    return [depl_c, depl_t, depl_tf, depl_c_s, depl_t_s, depl_tf_s, depl_f_s]


# %%
def time_series_depl(product, dates):
    list_depl = []
    for date in dates:
        new_mat = perc_mat(get_depl_mat(product, date))
        new_row = summ_depl(new_mat)
        list_depl = list_depl+[new_row]
    df_depl = pd.DataFrame(list_depl, columns=['Depl_Cancel', 'Depl_Trade',\
        'Depl_Trade+Fill', 'DC same-oppo', 'DT same-oppo', 'DT+F same-oppo',\
        'Fill same-oppo'], index=dates)
    return df_depl


# %%
def avg_perc_mat_2(product, dates):
    date_count = 1
    result = perc_mat(get_depl_mat(product, dates[date_count-1]))
    for date in dates[1:]:
        date_count += 1
        result += (perc_mat(get_depl_mat(product, dates[date_count-1]))-result)/date_count
    return result


# %% {"jupyter": {"outputs_hidden": true}}
get_depl_mat('DOL', '2017-01-03')

# %% {"jupyter": {"outputs_hidden": true}}
get_depl_mat('WDO', '2017-01-03')

# %% {"jupyter": {"outputs_hidden": true}}
perc_mat(get_depl_mat('DOL', '2017-01-03'))

# %%
DOL_DEPL_TS = time_series_depl('DOL', IMBAL_DATES)

# %%
DOL_DEPL_TS['eta1'] = DOL_ETA
DOL_DEPL_TS['Product'] = 'DOL'
DOL_DEPL_TS.index.names=['Dates']

# %%
WDO_DEPL_TS = time_series_depl('WDO', IMBAL_DATES)

# %%
WDO_DEPL_TS['eta1'] = WDO_ETA
WDO_DEPL_TS['Product'] = 'WDO'
WDO_DEPL_TS.index.names=['Dates']

# %%
ALL_DEPL_TS = pd.concat([WDO_DEPL_TS.reset_index(), DOL_DEPL_TS.reset_index()], sort=False)

# %%
DOL_DEPL_TS_MI = DOL_DEPL_TS.copy().drop(columns=['Product'])
DOL_DEPL_TS_MI.columns = pd.MultiIndex.from_product([['DOL'], DOL_DEPL_TS_MI.columns], names=['Product', 'Event'])

# %%
WDO_DEPL_TS_MI = WDO_DEPL_TS.copy().drop(columns=['Product'])
WDO_DEPL_TS_MI.columns = pd.MultiIndex.from_product([['WDO'], WDO_DEPL_TS_MI.columns], names=['Product', 'Event'])

# %%
ALL_DEPL_TS_MI = pd.concat([WDO_DEPL_TS_MI, DOL_DEPL_TS_MI], axis=1)

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='Depl_Cancel', hue='Product',\
           data=ALL_DEPL_TS);
plt.title('Depl_Cancel and $\eta$ for DOL and WDO');

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='DC same-oppo', hue='Product',\
           data=ALL_DEPL_TS);
plt.title('DC same-oppo and $\eta$ for DOL and WDO');

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='DT same-oppo', hue='Product',\
           data=ALL_DEPL_TS);
plt.title('DT same-oppo and $\eta$ for DOL and WDO');

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='DT+F same-oppo', hue='Product',\
           data=ALL_DEPL_TS);
plt.title('DT+F same-oppo and $\eta$ for DOL and WDO');

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(x='eta1', y='Fill same-oppo', hue='Product',\
           data=ALL_DEPL_TS);
plt.title('Filled: Same - Opposite and $\eta$ for DOL and WDO');

# %%
ALL_DEPL_TS_MI.loc[slice(None),(slice(None),'Depl_Cancel')]\
    .plot(figsize=(9, 6), title='Depleted by Cancel');

# %%
ALL_DEPL_TS_MI.loc[slice(None),(slice(None),'DC same-oppo')]\
    .plot(figsize=(9, 6), title='Depleted by Cancel: Same - Opposite');

# %%
ALL_DEPL_TS_MI.loc[slice(None),(slice(None),'Depl_Trade')]\
    .plot(figsize=(9, 6), title='Depleted by Trade');

# %%
ALL_DEPL_TS_MI.loc[slice(None),(slice(None),'DT same-oppo')]\
    .plot(figsize=(9, 6), title='Depleted by Trade: Same - Opposite');

# %%
ALL_DEPL_TS_MI.loc[slice(None),(slice(None),'DT+F same-oppo')]\
    .plot(figsize=(9, 6), title='Depleted by Trade+Filled: Same - Opposite');

# %%
ALL_DEPL_TS_MI.loc[slice(None),(slice(None),'Fill same-oppo')]\
    .plot(figsize=(9, 6), title='Filled: Same - Opposite');

# %%
perc_mat(get_depl_mat('WDO', '2017-01-03'))

# %%
perc_mat(get_depl_mat('DOL', '2017-01-04'))

# %%
perc_mat(get_depl_mat('WDO', '2017-01-04'))

# %%
np.round(avg_perc_mat_2('DOL', IMBAL_DATES),2)

# %%
np.round(avg_perc_mat_2('WDO', IMBAL_DATES),2)

# %%
np.round(avg_perc_mat_2('DOL', IMBAL_DATES),0)

# %%
np.round(avg_perc_mat_2('WDO', IMBAL_DATES),0)

# %% [markdown]
# ## Dataframe of price changes by k

# %%
DF_STATS_MOVES = pd.DataFrame()
for j in range(len(FILESDOL2017)):
    new_row = pd.read_hdf(PATHOUT+FILESDOL2017[j][:-4]+'_CAticks.h5')
    new_row['Dates'] = DATES_DOL[j].strftime('%Y-%m-%d')
    new_row['Product'] = FILESDOL2017[j][:3]
    new_row['Contract'] = FILESDOL2017[j][3:6]
    DF_STATS_MOVES = DF_STATS_MOVES.append(new_row)
for j in range(len(FILESWDO2017)):
    new_row = pd.read_hdf(PATHOUT+FILESWDO2017[j][:-4]+'_CAticks.h5')
    new_row['Dates'] = DATES_WDO[j].strftime('%Y-%m-%d')
    new_row['Product'] = FILESWDO2017[j][:3]
    new_row['Contract'] = FILESWDO2017[j][3:6]
    DF_STATS_MOVES = DF_STATS_MOVES.append(new_row)
DF_STATS_MOVES.reset_index()
DF_STATS_MOVES.set_index(['Product', 'Contract', 'Dates'], inplace=True)
DF_STATS_MOVES.sort_index(inplace=True)


# %%
def get_moves_mat(product, contract, date):
    mat_values = DF_STATS_MOVES.loc[product, contract, date].values
    mat_df = pd.DataFrame(mat_values,\
        columns=['Move in ticks', 'lambda',\
            'Continuations', 'Alternations', 'u', 'eta'],)
    return mat_df


# %%
def get_lambda1_mat():
    sub_mat = DF_STATS_MOVES.copy().reset_index()
    sub_mat = sub_mat[sub_mat['Li']==1]\
        [['Product', 'Contract', 'Dates', 'lamb']]
    return sub_mat


# %%
get_moves_mat('DOL','G17','2017-01-03')

# %%
get_moves_mat('WDO','G17','2017-01-03')

# %%
plt.figure(figsize=(9,9))
sns.lineplot(x='Dates', y='lamb', hue='Product', data=get_lambda1_mat());
plt.xticks(rotation=90);

# %%
#DOL_MEAN_COST[DOL_MEAN_COST['Trade Qty']<=300].plot.scatter(x='Trade Qty', y='Avg_Cost');

# %%
