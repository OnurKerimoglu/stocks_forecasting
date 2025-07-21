import os

import kaggle
import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

def load_raw_data(datapath, user, datasetname):
    print(f'rootpath: {datapath}')

    os.makedirs(datapath, exist_ok=True)
    if not os.path.exists(os.path.join(datapath, datasetname)):
        kaggle.api.dataset_download_files(
            dataset=f'{user}/{datasetname}',
            path=datapath,
            unzip=True)
    else:
        print('Raw data already found in location {}'.format(datapath))

    raw_fpath = os.listdir(os.path.join(datapath, datasetname))[0]
    raw_fpath_full = os.path.join(datapath, datasetname, raw_fpath)

    print(f'reading raw data from: {raw_fpath_full}')
    df_raw = pd.read_csv(raw_fpath_full)
    return df_raw

def clean_raw_data(df_raw):
    df_clean = df_raw.copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], utc=True).dt.tz_convert(None)
    df_clean.drop_duplicates(subset=['Date', 'Ticker'], keep='first', inplace=True)
    df_clean = df_clean[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
    df_clean['returns'] = df_clean['Close'].pct_change()
    df_clean.dropna(inplace=True)
    return df_clean

def sample_tickers_dates(df_clean, tickers=None, startdate=None, clean_sample_fpath_full=None):
    if tickers is None:
        df_clean_sample = df_clean.copy()
    else:
        print(f'Sampling tickers: {tickers}')
        df_clean_sample = df_clean[
            (df_clean['Ticker'].isin(tickers))
        ].copy()
    if startdate is not None:
        print(f'Sampling from start date: {startdate}')
        df_clean_sample = df_clean_sample[df_clean_sample['Date'] >= startdate].copy()
    # df_clean_sample.Date = pd.to_datetime(df_clean_sample['Date'])
    # print(f'sample shape: {df_clean_sample.shape}')
    # df_clean_sample.sort_values('Date', ascending=True).head()
    if clean_sample_fpath_full is not None:
        df_clean_sample.to_csv(clean_sample_fpath_full, index=False)
        print(f'Wrote clean sample to: {clean_sample_fpath_full}')
    df_clean_sample.sort_values(['Ticker', 'Date'], inplace=True)
    return df_clean_sample

def split_train_test(df, train_ratio=0.8):
    # df.drop(columns=['Close'], inplace=True)
    all_days = pd.Series(df.index.sort_values().values)
    split_date = all_days[int(len(all_days) * train_ratio)]
    train_df = df[df.index < split_date].copy()
    test_df  = df[df.index >=  split_date].copy()
    print(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')
    return train_df, test_df

def split_train_test_panel(df: pd.DataFrame,
                           train_ratio: float,
                           date_col: str = 'Date'
                          ):
    """
    Splits a panel DataFrame into train/test by date, preserving all tickers
    and *not* shuffling. 
    
    Parameters
    ----------
    df : pd.DataFrame
        Your panel data, either indexed by a DatetimeIndex, or containing
        a date column (specified by date_col).
    train_ratio : float
        Fraction of unique dates to use for training (e.g. 0.8).
    date_col : str, optional
        Name of the column containing dates, if df.index is not datetime.
    
    Returns
    -------
    df_train, df_test : (pd.DataFrame, pd.DataFrame)
        Two DataFrames containing the first train_ratio of dates, and the
        remaining dates, respectively.
    """
    # Extract dates
    dates = pd.to_datetime(df[date_col])

    # Determine split boundary
    unique_dates = pd.Index(dates).unique().sort_values()
    n_train = int(len(unique_dates) * train_ratio)
    if n_train < 1 or n_train >= len(unique_dates):
        raise ValueError("train_ratio produces empty train or test set")
    split_date = unique_dates[n_train - 1]

    # Boolean mask and split
    mask = dates <= split_date
    df_train = df.loc[mask]
    df_test  = df.loc[~mask]

    return df_train, df_test

def build_features(df_in, lags=3):
    feats = []
    for ticker, grp in df_in.groupby('Ticker'):
        df = grp.sort_values('Date').copy()

        # Base features
        df['volatility']  = (df['High'] - df['Low']) / ((df['High'] + df['Low']) / 2)
        df['dollar_vol']  = df['Volume'] * ((df['Open'] + df['Close']) / 2)

        # AR features
        lag_feat_names = []
        for lag in range(1, lags + 1):
            feat_name = f'returns_lag{lag}'
            lag_feat_names.append(feat_name)
            df[feat_name] = df['returns'].shift(lag)

        # Moving Averages (SMA & EMA)
        for w in [10, 50]:
            df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
        for w in [12, 26]:
            df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
        
        # Indices
        # MACD, Signal & Histogram
        df['MACD']        = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands & Width
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_STD']    = df['Close'].rolling(window=20).std()
        df['BB_Upper']  = df['BB_Middle'] + 2 * df['BB_STD']
        df['BB_Lower']  = df['BB_Middle'] - 2 * df['BB_STD']
        df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # RSI (14)
        delta     = df['Close'].diff()
        up        = delta.clip(lower=0)
        down      = -delta.clip(upper=0)
        avg_gain  = up.rolling(window=14).mean()
        avg_loss  = down.rolling(window=14).mean()
        rs        = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # Average True Range (14)
        high_low        = df['High'] - df['Low']
        high_prev_close = (df['High'] - df['Close'].shift(1)).abs()
        low_prev_close  = (df['Low'] - df['Close'].shift(1)).abs()
        tr              = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        df['ATR_14']    = tr.rolling(window=14).mean()

        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # Rate of Change (10)
        df['ROC_10'] = df['Close'].pct_change(periods=10)

        # Volatility & Dollar‐Volume Moving Averages & Ratios
        for w in [10, 20]:
            df[f'volatility_ma{w}'] = df['volatility'].rolling(window=w).mean()
            df[f'dollar_vol_ma{w}'] = df['dollar_vol'].rolling(window=w).mean()

        df['volatility_ratio']   = df['volatility']   / df['volatility_ma20']
        df['dollar_vol_ratio']   = df['dollar_vol']   / df['dollar_vol_ma20']


        # Trend + seasonality
        df['Period'] = df['Date'].dt.to_period('D')
        df = df.set_index('Period')
        dp = DeterministicProcess(
            index=df.index,
            constant=False, 
            order=0,  # no trend 
            seasonal=False,  # no additional seasonality terms
            additional_terms=[
                # CalendarFourier(freq='YE', order=1),
                # CalendarFourier(freq='QE', order=1),
                CalendarFourier(freq='ME', order=1),
                CalendarFourier(freq='W',  order=1)
            ]
        )
        tf = dp.in_sample()
        
        # Select features
        feature_cols = ['Ticker', 'Date', 'returns'] + lag_feat_names + [
            'volatility', 'dollar_vol',
            'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
            'RSI_14', 'ATR_14', 'OBV', 'ROC_10',
            'volatility_ma10', 'volatility_ma20',
            'dollar_vol_ma10', 'dollar_vol_ma20',
            'volatility_ratio', 'dollar_vol_ratio']
        df_feat = df[feature_cols]

        # Merge and reset index
        df_feat = df_feat.reset_index().drop(columns=['Period'])
        merged = pd.concat([df_feat.reset_index(drop=True), tf.reset_index(drop=True)], axis=1)

        # Drop rows with any NaNs
        merged = merged.dropna().reset_index(drop=True)

        feats.append(merged)
    # Concatenate all ticker dataframes
    df_out = pd.concat(feats, ignore_index=True)
    
    # We should standard-scale features expressed in price/volume units
    # while passing through already‐normalized or bounded ones.
    scale_features = lag_feat_names + [
        'volatility', 'dollar_vol',
        'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Middle', 'BB_Upper', 'BB_Lower',
        'ATR_14', 'OBV', 'ROC_10',
        'volatility_ma10', 'volatility_ma20',
        'dollar_vol_ma10', 'dollar_vol_ma20'
    ]
    return df_out, scale_features

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

def make_multistep_target(y, steps):
    y_multi = pd.concat(
        {f'y_step_{i + 1}': y.shift(-i)
         for i in range(steps)},
        axis=1
        )
    y_multi.dropna(inplace=True)
    return y_multi

def create_X_y_multistep(df_all, steps=5, target='Returns'):
    y_list = []
    X_list = []
    # loop over tickers to create multistep targets
    for ticker, grp in df_all.groupby('Ticker'):
        df = grp.sort_values('Date').copy()
        y = df[target]
        y_multi = make_multistep_target(y, steps=steps).dropna()
        X = df.drop(columns=[target]) # TODO: don't drop, use ticker for multi-indexing
        # Shifting has created indexes that don't match. Only keep times for
        # which we have both targets and features.
        y_multi, X = y_multi.align(X, join='inner', axis=0)
        # Add Ticker and Date, which will be used as indices later
        y_multi['Ticker'] = ticker
        y_multi['Date'] = X['Date']
        # check whether anything left from X and y_multi after droppping Nas
        if y_multi.shape[0] == 0 or X.shape[0] == 0:
            print(f"For ticker: {ticker}, no data left after dropping NaNs.")
        else:
            y_list.append(y_multi)
            X_list.append(X)
    if len(y_list) == 0 or len(X_list) == 0:
        raise ValueError("No data left after processing. Check your input data and parameters.")
    else:
        y_multi_all = pd.concat(y_list)
        X_all = pd.concat(X_list)
        print(f'X shape: {X_all.shape}, y_multi shape: {y_multi_all.shape}')
        X_all.set_index(['Ticker', 'Date'], inplace=True)
        y_multi_all.set_index(['Ticker', 'Date'], inplace=True)
        return X_all, y_multi_all
