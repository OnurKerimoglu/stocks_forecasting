import os

import kaggle
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing

def download_stock_price_prediction_challenge_data(rootpath):
    datapath = os.path.join(rootpath, "data")
    os.makedirs(datapath, exist_ok=True)
    competition="stock-price-prediction-challenge"

    os.makedirs(datapath, exist_ok=True)
    zipfpath = os.path.join(datapath, 'stock-price-prediction-challenge.zip')
    if not os.path.exists(zipfpath):
        print("Raw data was not found in location {}, downloading".format(zipfpath))
        kaggle.api.competition_download_cli(competition=competition, path=datapath)
    else:
        print("Raw data already found in location {}".format(zipfpath))

    # unzip the file
    dirfpath = os.path.join(datapath, 'stock-price-prediction-challenge')
    if not os.path.exists(dirfpath):
        print("Unzipping raw data")
        os.system(f"unzip {zipfpath} -d {dirfpath}")
    else:
        print("Raw data already unzipped")

    return dirfpath

def get_ticker_df(ticker_name, test_path, train_stocks_path):
    if 'test' in ticker_name:
        i = int(ticker_name.split('_')[1])
        assert i>=1 and i<=5
        fpath = os.path.join(test_path, f"test_{i}.csv")
    else:
        fpath = os.path.join(train_stocks_path, f"{ticker_name}.csv")
    if not os.path.exists(fpath):
        raise Exception(f"File {fpath} does not exist")
    df = pd.read_csv(fpath)
    return df

def add_indice_features(df_ticker, target, train_indices_path):
    # Read index data and merge in a single dataframe
    indices = {
        "dj": "Dow_Jones.csv",
        "nasdaq": "NASDAQ.csv",
        "SP500": "SP500.csv"
    }
    # merged_df = test_dfs[0][['Date', f'return_1', f'close_1', f'volume_1']]
    merged_index_df = df_ticker 
    for i, (key, filename) in enumerate(indices.items()):
        index_df = pd.read_csv(os.path.join(train_indices_path, filename))
        index_df = index_df.rename(columns={f"{target}": f"{target}_{key}"})
        merged_index_df = pd.merge(merged_index_df, index_df[['Date', f'{target}_{key}']], on='Date', how='left')
    # print(f'shape merged_index_df: {merged_index_df.shape}')
    merged_index_df.head()
    return merged_index_df

def prepare_features(df, target, beta_window=10, ma_windows=[10, 20, 60], ewm_alpha=[0.1, 0.3, 0], lags=1, out_len=500):
    extended_len = out_len + max(ma_windows)-1
    df = df[-extended_len:].copy()
    # print(f'shape of extended df: {df.shape}')
    
    # df['Adjusted_Yesterday'] = df['Adjusted'].shift(1)
    # df['rel_return'] = (df['Adjusted'] - df['Adjusted_Yesterday']) / df['Adjusted_Yesterday']
    df['Volatility'] = (df['High'] - df['Low']) / ((df['High'] + df['Low']) / 2)
    df['Dollar_Vol'] = df['Volume'] * ((df['Open'] + df['Close']) / 2)

    for lag in range(1, lags + 1):
        df[f'{target}_lag{lag}'] = df[target].shift(lag)

    for window in ma_windows:
        df[f'{target}_MA{window}'] = df[target].rolling(window=window).mean()

    # exponential smoothing
    for alpha in ewm_alpha:
        if alpha == 0:
            fit = SimpleExpSmoothing(df[target], initialization_method="estimated").fit()
            print(f'Optimal alpha for exponential smoothing: {fit.params["smoothing_level"]}')
            df[f'{target}_EWMopt'] = fit.fittedvalues
        else:
            df[f'{target}_EWM{alpha}'.replace('.', '')] = df[target].ewm(alpha=alpha).mean()
    

    df['Close_ROC10'] = df['Close'].pct_change(periods=10)

    df.index = pd.to_datetime(df['Date'])

    # rolling correlations with indices
    for index_key in ['dj', 'nasdaq', 'SP500']:
        returns_index_col = f'{target}_{index_key}'
        
        rolling_cov = df[[target, returns_index_col]].rolling(window=beta_window).cov()
        
        cov = rolling_cov.loc[
            rolling_cov.index.get_level_values(1) == returns_index_col, target
            ].reset_index(drop=True)
        cov.index = df.index

        var = df[returns_index_col].rolling(window=beta_window).var()
        var.index = df.index
        
        df[f'Beta{beta_window}_{index_key}'] = cov / var

    cols2drop = ['Date', 'Open', 'High', 'Low', 'Volume', 'Close', 'Adjusted']
    if 'Ticker' in df.columns:
        cols2drop.append('Ticker')
    df.drop(columns=cols2drop, inplace=True)
    df.dropna(inplace=True)
    print(f'shape of final df: {df.shape}')

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