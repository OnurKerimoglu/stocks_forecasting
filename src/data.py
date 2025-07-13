import os

import numpy as np
import pandas as pd

def load_raw_data(rootpath):
    print(f'rootpath: {rootpath}')
    datasetname = 'world-stock-prices-daily-updating'
    datapath = os.path.join(rootpath, 'data')

    raw_fpath = os.listdir(os.path.join(datapath, datasetname))[0]
    raw_fpath_full = os.path.join(datapath, datasetname, raw_fpath)

    print(f'reading raw data from: {raw_fpath_full}')
    df_raw = pd.read_csv(raw_fpath_full)
    return df_raw

def clean_raw_data(df_raw):
    df_clean = df_raw.copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], utc=True).dt.tz_convert(None)
    # df_clean['Date'] = df_clean['Date'].dt.date
    df_clean.drop_duplicates(subset=['Date', 'Ticker'], keep='first', inplace=True)
    df_clean = df_clean[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
    return df_clean

def prepare_base_features(df):
    df['rel_return'] = (df['Close'] - df['Open']) / df['Open']
    df['volatility'] = (df['High'] - df['Low']) / ((df['High'] + df['Low']) / 2)
    df['dollar_vol'] = df['Volume'] * ((df['Open'] + df['Close']) / 2)
    df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
    return df

# train test split
def split_df(df, train_ratio=0.8):
    # df.drop(columns=['Close'], inplace=True)
    all_days = pd.Series(df.index.sort_values().values)
    split_date = all_days[int(len(all_days) * train_ratio)]
    train_df = df[df.index < split_date].copy()
    test_df  = df[df.index >=  split_date].copy()
    print(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')
    return train_df, test_df

def split_features_targets(X1, X2, y, train_ratio=0.8):
    all_days = pd.Series(y.index.sort_values().values)
    split_date = all_days[int(len(all_days) * train_ratio)]
    X1_train = X1[X1.index < split_date].copy()
    X1_test  = X1[X1.index >=  split_date].copy()
    print(f'X1 Train shape: {X1_train.shape}, Test shape: {X1_test.shape}')
    X2_train = X2[X2.index < split_date].copy()
    X2_test  = X2[X2.index >=  split_date].copy()
    print(f'X2 Train shape: {X2_train.shape}, Test shape: {X2_test.shape}')
    y_train = y[y.index < split_date].copy()
    y_test  = y[y.index >=  split_date].copy()
    print(f'y Train shape: {y_train.shape}, Test shape: {y_test.shape}')

    return X1_train, X2_train, y_train, X1_test, X2_test, y_test

def extract_ticker(df_clean, ticker, requiredrecords=500, datapath=None, write=False,):
    if type(ticker) is not str:
        raise TypeError('ticker must be a string')
    print(f'Extracting for ticker: {ticker}')
    df_clean_sample = df_clean[
        (df_clean['Ticker'].isin([ticker])) #& 
        # (df_clean['Date'] >= date(2025, 1, 1))
        ].copy()
    # set date as index
    df_clean_sample.set_index('Date', inplace=True)
    # df_clean_sample = df_clean_sample.resample('D').mean()
    df_clean_sample.index = df_clean_sample.index.normalize()
    # drop unnecessary coluns
    df_clean_sample.drop(columns=['Ticker'], inplace=True)
    # drop rows with nans
    df_clean_sample = df_clean_sample.dropna()
    df_clean_sample = df_clean_sample.sort_values('Date', ascending=False)
    # take only the requested number of records from the latest period
    df_clean_sample = df_clean_sample.head(requiredrecords)
    print(f'Extracted sample shape: {df_clean_sample.shape}')
    
    if write:
        clean_sample_fpath_full = os.path.join(datapath, f'clean_sample_{ticker}.csv')
        df_clean_sample.to_csv(clean_sample_fpath_full, index=False)
        print(f'Wrote sample to: {clean_sample_fpath_full}')
    return df_clean_sample

def make_lags(ts, lags, lead_time=1, name='y'):
    return pd.concat(
        {
            f'{name}_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)

def make_leads(ts, leads, name='y'):
    return pd.concat(
        {
            f'{name}_lead_{i}': ts.shift(-i)
            for i in reversed(range(leads))
        },
        axis=1)

def make_multistep_target(ts, steps, reverse=False):
    shifts = reversed(range(steps)) if reverse else range(steps)
    return pd.concat({f'y_step_{i + 1}': ts.shift(-i) for i in shifts}, axis=1)


def create_multistep_example(n, steps, lags, lead_time=1):
    ts = pd.Series(
        np.arange(n),
        index=pd.period_range(start='2010', freq='A', periods=n, name='Year'),
        dtype=pd.Int8Dtype,
    )
    X = make_lags(ts, lags, lead_time)
    y = make_multistep_target(ts, steps, reverse=True)
    data = pd.concat({'Targets': y, 'Features': X}, axis=1)
    data = data.style.set_properties(['Targets'], **{'background-color': 'LavenderBlush'}) \
                     .set_properties(['Features'], **{'background-color': 'Lavender'})
    return data


def load_multistep_data():
    df1 = create_multistep_example(10, steps=1, lags=3, lead_time=1)
    df2 = create_multistep_example(10, steps=3, lags=4, lead_time=2)
    df3 = create_multistep_example(10, steps=3, lags=4, lead_time=1)
    return [df1, df2, df3]