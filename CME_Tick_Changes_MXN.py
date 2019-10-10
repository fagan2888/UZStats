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
CURR = 'MXN'

# %%
PATH_PRIOR = PATHPROJ+CURR+'/prior/'
PATH_AFTER = PATHPROJ+CURR+'/after/'
URL_1 = CURR+'/prior/'
URL_2 = CURR+'/after/'
PATH_PRIOR = URL_ROOT+URL_1
PATH_AFTER = URL_ROOT+URL_2

# %%
TRADING_HOURS = 8

# %%
TICK_PRIOR = 12.5
TICK_AFTER = 5.0

# %%
PRIOR_CDATES_LIST = [['6MM4', '20140317'], ['6MM4', '20140318'], ['6MM4', '20140319'],\
    ['6MM4', '20140320'], ['6MM4', '20140321'], ['6MM4', '20140324'], ['6MM4', '20140325'],\
    ['6MM4', '20140326'], ['6MM4', '20140327'], ['6MM4', '20140328'], ['6MM4', '20140331'],\
    ['6MM4', '20140401'], ['6MM4', '20140402'], ['6MM4', '20140403'], ['6MM4', '20140404'],\
    ['6MM4', '20140407'], ['6MM4', '20140408'], ['6MM4', '20140409'], ['6MM4', '20140410'],\
    ['6MM4', '20140411'], ['6MM4', '20140414'], ['6MM4', '20140415'], ['6MM4', '20140416'],\
    ['6MM4', '20140417'], ['6MM4', '20140421'], ['6MM4', '20140422'], ['6MM4', '20140423'],\
    ['6MM4', '20140424'], ['6MM4', '20140425'], ['6MM4', '20140428'], ['6MM4', '20140429'],\
    ['6MM4', '20140430'], ['6MM4', '20140501'], ['6MM4', '20140502'], ['6MM4', '20140505'],\
    ['6MM4', '20140506'], ['6MM4', '20140507'], ['6MM4', '20140508'], ['6MM4', '20140509'],\
    ['6MM4', '20140512'], ['6MM4', '20140513'], ['6MM4', '20140514'], ['6MM4', '20140515'],\
    ['6MM4', '20140516'], ['6MM4', '20140519'], ['6MM4', '20140520'], ['6MM4', '20140521'],\
    ['6MM4', '20140522'], ['6MM4', '20140523'], ['6MM4', '20140526'], ['6MM4', '20140527'],\
    ['6MM4', '20140528'], ['6MM4', '20140529'], ['6MM4', '20140530'], ['6MM4', '20140602'],\
    ['6MM4', '20140603'], ['6MM4', '20140604'], ['6MM4', '20140605'], ['6MM4', '20140606'],\
    ['6MM4', '20140609'], ['6MM4', '20140610'], ['6MM4', '20140611'], ['6MM4', '20140612'],\
    ['6MM4', '20140613'], ['6MU4', '20140616'], ['6MU4', '20140617'], ['6MU4', '20140618'],\
    ['6MU4', '20140619'], ['6MU4', '20140620'], ['6MU4', '20140623'], ['6MU4', '20140624'],\
    ['6MU4', '20140625'], ['6MU4', '20140626'], ['6MU4', '20140627'], ['6MU4', '20140630'],\
    ['6MU4', '20140701'], ['6MU4', '20140702'], ['6MU4', '20140703'], ['6MU4', '20140704'],\
    ['6MU4', '20140707'], ['6MU4', '20140708'], ['6MU4', '20140709'], ['6MU4', '20140710'],\
    ['6MU4', '20140711']]

# %%
AFTER_CDATES_LIST = [['6MU4', '20140715'], ['6MU4', '20140716'], ['6MU4', '20140717'],\
    ['6MU4', '20140718'], ['6MU4', '20140721'], ['6MU4', '20140722'], ['6MU4', '20140723'],\
    ['6MU4', '20140724'], ['6MU4', '20140725'], ['6MU4', '20140728'], ['6MU4', '20140729'],\
    ['6MU4', '20140730'], ['6MU4', '20140731'], ['6MU4', '20140801'], ['6MU4', '20140804'],\
    ['6MU4', '20140805'], ['6MU4', '20140806'], ['6MU4', '20140807'], ['6MU4', '20140808'],\
    ['6MU4', '20140811'], ['6MU4', '20140812'], ['6MU4', '20140813'], ['6MU4', '20140814'],\
    ['6MU4', '20140815'], ['6MU4', '20140818'], ['6MU4', '20140819'], ['6MU4', '20140820'],\
    ['6MU4', '20140821'], ['6MU4', '20140822'], ['6MU4', '20140825'], ['6MU4', '20140826'],\
    ['6MU4', '20140827'], ['6MU4', '20140828'], ['6MU4', '20140829'], ['6MU4', '20140901'],\
    ['6MU4', '20140902'], ['6MU4', '20140903'], ['6MU4', '20140904'], ['6MU4', '20140905'],\
    ['6MU4', '20140908'], ['6MU4', '20140909'], ['6MU4', '20140910'], ['6MU4', '20140911'],\
    ['6MU4', '20140912'], ['6MZ4', '20140915'], ['6MZ4', '20140916'], ['6MZ4', '20140917'],\
    ['6MZ4', '20140918'], ['6MZ4', '20140919'], ['6MZ4', '20140922'], ['6MZ4', '20140923'],\
    ['6MZ4', '20140924'], ['6MZ4', '20140925'], ['6MZ4', '20140926'], ['6MZ4', '20140929'],\
    ['6MZ4', '20140930'], ['6MZ4', '20141001'], ['6MZ4', '20141002'], ['6MZ4', '20141003'],\
    ['6MZ4', '20141006'], ['6MZ4', '20141007'], ['6MZ4', '20141008'], ['6MZ4', '20141009'],\
    ['6MZ4', '20141010'], ['6MZ4', '20141013'], ['6MZ4', '20141014'], ['6MZ4', '20141015'],\
    ['6MZ4', '20141016'], ['6MZ4', '20141017'], ['6MZ4', '20141020'], ['6MZ4', '20141021'],\
    ['6MZ4', '20141022'], ['6MZ4', '20141023'], ['6MZ4', '20141024'], ['6MZ4', '20141027'],\
    ['6MZ4', '20141028'], ['6MZ4', '20141029'], ['6MZ4', '20141030'], ['6MZ4', '20141031'],\
    ['6MZ4', '20141103'], ['6MZ4', '20141104'], ['6MZ4', '20141105'], ['6MZ4', '20141106'],\
    ['6MZ4', '20141107']]

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
    'Relative predictive power of imbalance : '+CURR, -1.0, 3.0, 50)

# %%
cme.time_series_hist_plot(ABSDEPL_STATS_TS, 'Depl_Cancel',\
    'Depl_Cancel : '+CURR, 0.0, 20000.0, 50)

# %%
cme.time_series_hist_plot(ABSDEPL_STATS_TS, 'Depl_Trades',\
    'Depl_Trades : '+CURR, 0.0, 15000.0, 50)

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
    'Depl_Cancel : '+CURR, 0, 50, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DC same-oppo',\
    'DC same-oppo : '+CURR, 0, 50, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Depl_Trade',\
    'Depl_Trade : '+CURR, 0, 40, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DT same-oppo',\
    'DT same-oppo : '+CURR, -2, 25, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'DT+F same-oppo',\
    'DT+F same-oppo : '+CURR, -2, 10, 50)

# %%
cme.time_series_hist_plot(DEPL_STATS_TS, 'Fill same-oppo',\
    'Filled : Same - Opposite : '+CURR, -25, 40, 50)

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
    'Spread in Ticks : '+CURR, 0.9, 2.5, 50)

# %%
cme.twspr_plot_USD(OB_UZ_STATS, CURR)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'eta1',\
    '$\eta$ : '+CURR, 0, 0.55, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'chgavg',\
    'Average Price Change : '+CURR, 9, 30, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'rvxe',\
    'Estimated Volatility of Efficient Prices : '+CURR, 0, 0.010, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'ndfpr',\
    'Number of Price Changes : '+CURR, 0, 4000, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'M',\
    'Number of Trades : '+CURR, 0, 20000, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS, 'Volume',\
    'Volume : '+CURR, 0, 100000, 50)

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
    'Level 1 Bid Average Amount : '+CURR, 0, 250, 50)

# %%
cme.time_series_hist_plot(OB_UZ_STATS_SPREADS, 'ask1qty',\
    'Level 1 Ask Average Amount : '+CURR, 0, 250, 50)

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
