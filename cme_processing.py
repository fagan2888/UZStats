# The Robert and Rosenbaum Uncertainty Zones model
# An application to FX Futures at CME
# Implementation by
# Marcos Costa Santos Carreira (École Polytechnique - CMAP)
# and
# Florian Huchedé (CME)
# Aug-2019

# Import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import glob

# Load files

def list_files(path):
    files_list = [f for f in glob.glob(path+'*_CAticks.csv', recursive=True)]
    files_list.sort()
    return [f[67:-12].split('_') for f in files_list]

def process_files(path, files_list, status, tick_value):
    full_path = path
    cdates = pd.DataFrame(files_list, columns=['Contract', 'Date'])
    cdates['Status'] = status
    cdates['Tick'] = tick_value
    files_CAticks = [full_path+cdates['Contract'].iloc[j]+\
        '_'+cdates['Date'].iloc[j]+'_CAticks.csv'\
        for j in range(len(cdates))]
    files_COSTtrades = [full_path+cdates['Contract'].iloc[j]+\
        '_'+cdates['Date'].iloc[j]+'_COSTtrades.csv'\
        for j in range(len(cdates))]
    files_OBstats = [full_path+cdates['Contract'].iloc[j]+\
        '_'+cdates['Date'].iloc[j]+'_OBstats.csv'\
        for j in range(len(cdates))]
    files_OTtrans = [full_path+cdates['Contract'].iloc[j]+\
        '_'+cdates['Date'].iloc[j]+'_OTtrans.csv'\
        for j in range(len(cdates))]
    files_RDFtrans = [full_path+cdates['Contract'].iloc[j]+\
        '_'+cdates['Date'].iloc[j]+'_RDFtrans.csv'\
        for j in range(len(cdates))]
    files_UZstats = [full_path+cdates['Contract'].iloc[j]+\
        '_'+cdates['Date'].iloc[j]+'_UZstats.csv'\
        for j in range(len(cdates))]
    return [cdates, files_CAticks, files_COSTtrades, files_OBstats,\
               files_OTtrans, files_RDFtrans, files_UZstats]

# OB and UZ Stats

def ob_uz_stats(cdates, files_OBstats, files_UZstats, files_CAticks,\
    trading_hours):
    df_stats = pd.DataFrame()
    for j in range(len(cdates)):
        new_row_1 = pd.read_csv(files_OBstats[j], index_col=0)
        new_row_2 = pd.read_csv(files_UZstats[j], index_col=0)
        new_row = pd.concat([new_row_1, new_row_2], axis=1, sort=False)
        new_row['lambda1'] = pd.read_csv(files_CAticks[j],\
            index_col=0).set_index('Li').loc[1]['lamb']
        df_stats = df_stats.append(new_row)
    df_stats.reset_index(drop=True, inplace=True)
    df_stats['Contract'] = cdates['Contract']
    df_stats['Date'] = pd.to_datetime(cdates['Date'])
    df_stats['Status'] = cdates['Status']
    df_stats['Tick'] = cdates['Tick']
    df_stats.set_index(['Date'], inplace=True)
    df_stats.sort_index(inplace=True)
    df_stats['ndfpr_pred'] = trading_hours*3600/df_stats['duration']
    # df_stats['ndfpr_pred'] = ((df_stats['rvxe']*df_stats['spot_avg']/\
    #     df_stats['Tick'])**2)/(2*df_stats['eta1'])
    df_stats['eta*alpha*sqrt(M)'] = df_stats['eta1']*df_stats['Tick']*\
        np.sqrt(df_stats['M'])
    df_stats['S*sqrt(M)'] = df_stats['twspr1']*df_stats['Tick']*\
        np.sqrt(df_stats['M'])
    df_stats['sigma'] = df_stats['rvxe']*df_stats['spot_avg']
    return df_stats

def table_mathieu(data_frame):
    return np.round(data_frame.reset_index().set_index(['Status'])\
        [['Tick', 'chgavg', 'ndfpr_pred', 'ndfpr', 'M', 'Volume', 'eta1',\
        'S1', 'lambda1', 'twspr1', 'duration', 'dt_avg', 'rvxe', 'spot_avg']].\
        groupby('Status').mean().sort_index(ascending=False), 5)

def table_mathieu_err(data_frame):
    return np.round(data_frame.reset_index().set_index(['Status'])\
        [['Tick', 'chgavg', 'ndfpr_pred', 'ndfpr', 'M', 'Volume', 'eta1',\
        'S1', 'lambda1', 'twspr1', 'duration', 'dt_avg', 'rvxe', 'spot_avg']].\
        groupby('Status').std().sort_index(ascending=False), 5)

# Imbalance Stats

def imbal_stats(cdates, files_OTtrans):
    df_stats = pd.DataFrame()
    for j in range(len(cdates)):
        new_row = pd.read_csv(files_OTtrans[j], index_col=0)
#         new_row['Contract'] = cdates['Contract'].loc[j]
        new_row['Dates'] = pd.to_datetime(cdates['Date'].loc[j])
#         new_row['Status'] = cdates['Status'].loc[j]
#         new_row['Tick'] = cdates['Tick'].loc[j]
        df_stats = df_stats.append(new_row)   
    df_stats.reset_index(drop=True, inplace=True)
    df_stats.set_index(['Dates'], inplace=True)
    df_stats.sort_index(inplace=True)
    return df_stats

def get_imbal_mat(data_frame, date):
    mat_values = data_frame.loc[date].values
    mat_df = pd.DataFrame(mat_values,\
        columns=['Trade_Bid', 'Imbal_Bid', 'Neutral', 'Imbal_Ask',\
            'Trade_Ask'],\
        index=['Trade_Bid', 'Imbal_Bid', 'Neutral', 'Imbal_Ask',\
            'Trade_Ask'])
    return mat_df

def perc_mat(mat):
    sum_mat = mat.to_numpy().sum()
    norm_mat = (100*mat/sum_mat)
    norm_mat['Total Cols'] = norm_mat.sum(axis=1)
    norm_mat.loc['Total Rows'] = norm_mat.sum(axis=0)
    return norm_mat

def perc_mat_r(mat, dec=2):
    sum_mat = mat.to_numpy().sum()
    norm_mat = (100*mat/sum_mat)
    norm_mat['Total Cols'] = norm_mat.sum(axis=1)
    norm_mat.loc['Total Rows'] = norm_mat.sum(axis=0)
    return norm_mat.round(dec)

def avg_perc_mat(data_frame, dates, dec=2):
    date_count = 1
    result = perc_mat(get_imbal_mat(data_frame, dates[date_count-1]))
    for date in dates[1:]:
        date_count += 1
        result += (perc_mat(get_imbal_mat(data_frame, dates[date_count-1]))-\
            result)/date_count
    return result.round(dec)

def summ_imbal(mat):
    pred_ib = (mat.loc['Imbal_Bid']['Trade_Bid']-\
        mat.loc['Imbal_Bid']['Trade_Ask'])
    pred_ia = (mat.loc['Imbal_Ask']['Trade_Ask']-\
        mat.loc['Imbal_Ask']['Trade_Bid'])
    pred_imb = (pred_ib+pred_ia)/2
    pred_ibr = pred_ib/mat.loc['Imbal_Bid']['Total Cols']*100
    pred_iar = pred_ia/mat.loc['Imbal_Ask']['Total Cols']*100
    pred_imbr = (pred_ibr+pred_iar)/2
    return [pred_imb, pred_imbr]

def time_series_imbal(data_frame, dates, status):
    list_imbal = []
    for date in dates:
        new_mat = perc_mat(get_imbal_mat(data_frame, date))
        new_row = summ_imbal(new_mat)
        list_imbal = list_imbal+[new_row]
    df_imbal = pd.DataFrame(list_imbal, columns=['Pred_Imbal',\
         'Pred_Imbal_Relat'], index=dates)
    df_imbal['Status'] = status
    return df_imbal

def summ_trade(mat):
    pred_tb = (mat.loc['Trade_Bid']['Trade_Bid']-\
        mat.loc['Trade_Bid']['Trade_Ask'])
    pred_ta = (mat.loc['Trade_Ask']['Trade_Ask']-\
        mat.loc['Trade_Ask']['Trade_Bid'])
    pred_trd = (pred_tb+pred_ta)/2
    pred_tbr = pred_tb/mat.loc['Trade_Bid']['Total Cols']*100
    pred_tar = pred_ta/mat.loc['Trade_Ask']['Total Cols']*100
    pred_trdr = (pred_tbr+pred_tar)/2
    return [pred_trd, pred_trdr]

def time_series_imbal_trd(data_frame, dates, status):
    list_imbal = []
    for date in dates:
        new_mat = perc_mat(get_imbal_mat(data_frame, date))
        new_row = summ_trade(new_mat)
        list_imbal = list_imbal+[new_row]
    df_imbal = pd.DataFrame(list_imbal, columns=['Pred_Trade',\
        'Pred_Trade_Relat'], index=dates)
    df_imbal['Status'] = status
    return df_imbal

# Depletion Stats

def depl_stats(cdates, files_RDFtrans):
    df_stats = pd.DataFrame()
    for j in range(len(cdates)):
        new_row = pd.read_csv(files_RDFtrans[j], index_col=0, header=[0,1])
#         new_row['Contract'] = cdates['Contract'].loc[j]
        new_row['Dates'] = pd.to_datetime(cdates['Date'].loc[j])
#         new_row['Status'] = cdates['Status'].loc[j]
#         new_row['Tick'] = cdates['Tick'].loc[j]
        df_stats = df_stats.append(new_row)   
    df_stats.reset_index(drop=True, inplace=True)
    df_stats.set_index(['Dates'], inplace=True)
    df_stats.sort_index(inplace=True)
    return df_stats

def get_depl_mat(data_frame, date):
    mat_values = data_frame.loc[date].values
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
        index=[' D C ', ' D T ', 'D T+F', '  F  '])
    return mat_df

def summ_depl(mat):
    depl_c = mat.loc[' D C ']['Total Cols'].sum()
    depl_t = mat.loc[' D T ']['Total Cols'].sum()
    depl_tf = mat.loc['D T+F']['Total Cols'].sum()
    depl_c_s = mat.loc[' D C ']['same', '  F  ']-\
        mat.loc[' D C ']['oppo', '  F  ']
    depl_t_s = mat.loc[' D T ']['same', '  F  ']-\
        mat.loc[' D T ']['oppo', '  F  ']
    depl_tf_s = mat.loc['D T+F']['same', '  F  ']-\
        mat.loc['D T+F']['oppo', '  F  ']
    depl_f_s = mat.loc['  F  ']['same'].sum()-mat.loc['  F  ']['oppo'].sum()
    return [depl_c, depl_t, depl_tf, depl_c_s, depl_t_s, depl_tf_s, depl_f_s]

def time_series_depl(data_frame, dates, status):
    list_depl = []
    for date in dates:
        new_mat = perc_mat(get_depl_mat(data_frame, date))
        new_row = summ_depl(new_mat)
        list_depl = list_depl+[new_row]
    df_depl = pd.DataFrame(list_depl, columns=['Depl_Cancel', 'Depl_Trade',\
        'Depl_Trade+Fill', 'DC same-oppo', 'DT same-oppo', 'DT+F same-oppo',\
        'Fill same-oppo'], index=dates)
    df_depl['Status'] = status
    return df_depl

def avg_perc_mat_2(data_frame, dates, dec=2):
    date_count = 1
    result = perc_mat(get_depl_mat(data_frame, dates[date_count-1]))
    for date in dates[1:]:
        date_count += 1
        result += (perc_mat(get_depl_mat(data_frame, dates[date_count-1]))-\
            result)/date_count
    return result.round(dec)

def summ_absdepl(mat):
    depl_c = mat.loc[' D C '].sum()
    depl_t = mat.loc[' D T '].sum()
    depl_tf = mat.loc['D T+F'].sum()
    return [depl_c, depl_t+depl_tf]

def time_series_absdepl(data_frame, dates, status):
    list_depl = []
    for date in dates:
        new_mat = get_depl_mat(data_frame, date)
        new_row = summ_absdepl(new_mat)
        list_depl = list_depl+[new_row]
    df_depl = pd.DataFrame(list_depl, columns=['Depl_Cancel', 'Depl_Trades'],\
        index=dates)
    df_depl['Status'] = status
    return df_depl

# Spread stats

def spread_stats(data):
    data_spreads = data.copy()[['ask12qty', 'ask12tomid', 'ask1qty',\
        'ask1tomid', 'bid12qty', 'bid12tomid', 'bid1qty', 'bid1tomid',\
        'twspr1', 'twspr2', 'chgavg', 'eta1', 'Tick', 'Status']]
    data_spreads['ask_adj_qty'] = np.where(data_spreads['Status']=='prior',\
        data_spreads['ask1qty'].copy(), data_spreads['ask12qty'].copy())
    data_spreads['bid_adj_qty'] = np.where(data_spreads['Status']=='prior',\
        data_spreads['bid1qty'].copy(), data_spreads['bid12qty'].copy())
    data_spreads['ask_adj_tomid'] = np.where(data_spreads['Status']=='prior',\
        data_spreads['ask1tomid']*data_spreads['Tick'],\
        data_spreads['ask12tomid']*data_spreads['Tick'])
    data_spreads['bid_adj_tomid'] = np.where(data_spreads['Status']=='prior',\
        data_spreads['bid1tomid']*data_spreads['Tick'],\
        data_spreads['bid12tomid']*data_spreads['Tick'])
    return data_spreads

# Cost Stats

def cost_stats(cdates, files_COSTtrades):
    df_stats = pd.DataFrame()
    for j in range(len(cdates)):
        new_row = pd.read_csv(files_COSTtrades[j], index_col=0)
#         new_row['Contract'] = cdates['Contract'].loc[j]
        new_row['Dates'] = pd.to_datetime(cdates['Date'].loc[j])
#         new_row['Status'] = cdates['Status'].loc[j]
#         new_row['Tick'] = cdates['Tick'].loc[j]
        df_stats = df_stats.append(new_row)   
    df_stats.reset_index(drop=True, inplace=True)
    df_stats.set_index(['Dates'], inplace=True)
    df_stats.sort_index(inplace=True)
    return df_stats

def cost_mean(data_frame, max_amount=200):
    data = data_frame.reset_index()
    data = data[data['Trade Qty'] <= max_amount]
    grouped = data.groupby(['Trade Qty']).mean()
    return grouped

# Linear regression

def lin_reg(data, independent, dependent, logdata=False):
    X = data[independent]
    Y = data[dependent]
    if logdata:
        X = np.log(X)
        Y = np.log(Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    #predictions = model.predict(X) 
    print_model = model.summary()
    print(print_model)

def lin_reg_params(data, independent, dependent, logdata=False):
    X = data[independent]
    Y = data[dependent]
    if logdata:
        X = np.log(X)
        Y = np.log(Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model.params

def lin_reg_rob(data, independent, dependent, logdata=False):
    X = data[independent]
    Y = data[dependent]
    if logdata:
        X = np.log(X)
        Y = np.log(Y)
    X = sm.add_constant(X)
    model = sm.RLM(Y, X).fit()
    #predictions = model.predict(X) 
    print_model = model.summary()
    print(print_model)

def lin_reg_rob_params(data, independent, dependent, logdata=False):
    X = data[independent]
    Y = data[dependent]
    if logdata:
        X = np.log(X)
        Y = np.log(Y)
    X = sm.add_constant(X)
    model = sm.RLM(Y, X).fit()
    return model.params

# Charts

def time_series_plot(database, field, title, y_min=0):
    df_plot = database[[field, 'Status']].copy()
    df_plot['prior'] = np.where(df_plot['Status'] == 'prior',\
        df_plot[field], np.nan)
    df_plot['after'] = np.where(df_plot['Status'] == 'after',\
        df_plot[field], np.nan)
    df_plot[['prior', 'after']].plot(figsize=(9,6), title=title,\
        marker='.', linestyle='None')
    plt.ylim(bottom=y_min)

def twspr_plot_USD(database, title):
    df_plot = database[['twspr1', 'Tick', 'Status']].copy()
    df_plot['twspr1'] = df_plot['twspr1']*df_plot['Tick']
    time_series_plot(df_plot, 'twspr1', 'Spread in USD : '+title)

def time_series_hist(database, field, title):
    df_plot = database[[field, 'Status']].copy()
    df_plot['prior'] = np.where(df_plot['Status'] == 'prior',\
        df_plot[field], np.nan)
    df_plot['after'] = np.where(df_plot['Status'] == 'after',\
        df_plot[field], np.nan)
    df_plot[['prior','after']].plot.hist(alpha=0.5, figsize=(9,6))
    plt.title(title)

def scatter_plot(database, field1, field2, title):
    plt.figure(figsize=(9, 6))
    plt.title(title)
    sns.scatterplot(x=field1, y=field2, hue='Status',\
            data=database);

def cloud1(database, title, robust=False):
    sns.lmplot(x='eta*alpha*sqrt(M)', y='sigma', hue='Status', data=database,\
              height=6, aspect=1.5, robust=robust);
    plt.title(r' Cloud ('
              r'$\eta\alpha\sqrt{M}, \sigma)$ : '+title);
    plt.xlabel(r'$\eta\alpha\sqrt{M}$');
    plt.ylabel(r'$\sigma$');

def cloud2(database, title, robust=False):
    sns.lmplot(x='p1*eta*alpha*sqrt(M)', y='sigma-p2*S*sqrt(M)', hue='Status',\
        data=database, height=6, aspect=1.5, robust=robust);
    plt.title(r' Cloud ('
              r'$p_{1}\eta\alpha\sqrt{M}, \sigma-p_{2}S\sqrt{M})$ : '+title);
    plt.xlabel(r'$p_{1}\eta\alpha\sqrt{M}$');
    plt.ylabel(r'$\sigma-p_{2}S\sqrt{M}$');

def regr_plot(database, field1, field2, title, robust=False):
    sns.lmplot(x=field1, y=field2, hue='Status', aspect=1, \
            height=7, data=database, robust=robust);
    plt.title(title)

# Eta prediction

def new_eta(old_tick, new_tick, old_eta, beta=1, lin_params=0):
    return (old_eta+lin_params)*((old_tick/new_tick)**(1-beta/2))-lin_params

new_eta_vect = np.vectorize(new_eta)

def plot_eta(old_tick, new_tick, old_eta, new_eta, old_eta_err, new_eta_err,\
    curr, beta=1, lin_params=0):
    x = np.linspace(0.5*new_tick, old_tick, 50)
    y = new_eta_vect(old_tick, x, old_eta, beta, lin_params)
    y_up = new_eta_vect(old_tick, x, old_eta+old_eta_err, beta, lin_params)
    y_down = new_eta_vect(old_tick, x, old_eta-old_eta_err, beta, lin_params)
    plt.figure(1, figsize=(9, 6))
    plt.title('$\eta$ prediction : '+curr)
    plt.xlabel('New tick size')
    plt.xlim((0.5*new_tick, old_tick))
    plt.ylim((0, 0.75))
    plt.xticks(np.arange(0.5*new_tick, old_tick, old_tick/10))
    plt.ylabel('New $\eta$')
    plt.plot(x, y)
    plt.fill_between(x, y_down, y_up, alpha=0.3)
    plt.plot([new_tick, new_tick], [0, new_eta], color='r')
    plt.plot([0, new_tick], [new_eta, new_eta], color='k')
    plt.plot([0, new_tick], [new_eta-new_eta_err, new_eta-new_eta_err],\
        color='g')
    plt.plot([0, new_tick], [new_eta+new_eta_err, new_eta+new_eta_err],\
        color='g')
