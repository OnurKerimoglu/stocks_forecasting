import os

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.models.functional import Functional as KerasFunctional
from keras.src.models.sequential import Sequential as KerasSequential
from xgboost import XGBRegressor

import data_challenge as data

def make_multistep_target(y, steps):
    y_multi = pd.concat(
        {f'y_step_{i + 1}': y.shift(-i)
         for i in range(steps)},
        axis=1
        )
    y_multi.dropna(inplace=True)
    return y_multi

def get_X_y_multistep(df, steps=11, target='Returns', forecast_horizon=1):
    y = df[target]
    y_multi = make_multistep_target(y, steps=steps).dropna()
    X = df.drop(columns=[target])
    # Shifting has created indexes that don't match. Only keep times for
    # which we have both targets and features.
    y_multi, X = y_multi.align(X, join='inner', axis=0)
    print(f'X shape: {X.shape}, y_multi shape: {y_multi.shape}')
    return X, y_multi

def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax

def evaluate_multistep(y_train, y_hat_train, y_test, y_hat_test, df, target):
    train_rmse = root_mean_squared_error(y_train, y_hat_train)
    test_rmse = root_mean_squared_error(y_test, y_hat_test)
    print((f"Train RMSE: {train_rmse:.5f}\n" f"Test RMSE: {test_rmse:.5f}"))

    plt.rc("figure", autolayout=True, figsize=(12, 6))
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=16,
        titlepad=10,
    )

    plot_params = dict(
        color="0.75",
        style=".-",
        markeredgecolor="0.25",
        markerfacecolor="0.25",
    )
    palette = dict(palette='husl', n_colors=64)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    ax1 = df[target][y_hat_train.index].plot(**plot_params, ax=ax1)
    ax1 = plot_multistep(y_hat_train, ax=ax1, palette_kwargs=palette)
    _ = ax1.legend([f'{target} (train)', 'Forecast'])
    ax2 = df[target][y_hat_test.index].plot(**plot_params, ax=ax2)
    ax2 = plot_multistep(y_hat_test, ax=ax2, palette_kwargs=palette)
    _ = ax2.legend([f'{target} (test)', 'Forecast'])

# use hyperparameters to initialize a new model
def initialize_XGBRegressorChain(params):
    base_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='auto',
        seed=42,
        **params
    )
    # Pack it into Regressor Chain to predict the next 10 days
    modelchain = RegressorChain(estimator=base_model)
    return modelchain

def make_forecast(df, estimator, target, columns):
    X_pred = df.iloc[[-1], :]
    X_pred = X_pred.drop(columns=[target])
    y_pred = estimator.predict(X_pred)
    y_pred = pd.DataFrame(y_pred, index=X_pred.index, columns=columns)
    return y_pred
 
def make_scaled_forecast_4sequence(df, estimator, target, columns, Xscaler, yscaler, window_size):
    X_pred = df.iloc[-window_size-1:, :]
    X_pred = X_pred.drop(columns=[target])
    X_pred_scaled = Xscaler.transform(X_pred)
    X_pred_scaled_seq, _ = make_sequences(X_pred_scaled, X_pred_scaled[:,0], window_size)
    y_pred_scaled = estimator.predict(X_pred_scaled_seq)
    y_pred = yscaler.inverse_transform(y_pred_scaled)
    y_pred = pd.DataFrame(y_pred, index=X_pred.iloc[[-1],:].index, columns=columns)
    return y_pred

def build_scalers(X_train, y_train, X_test, y_test):
    Xscaler = MinMaxScaler(feature_range=(0, 1))
    yscaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = Xscaler.fit_transform(X_train)
    X_test_scaled = Xscaler.transform(X_test)
    y_train_scaled = yscaler.fit_transform(y_train)
    y_test_scaled = yscaler.transform(y_test)
    return Xscaler, yscaler, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

def make_sequences(X, y, time_steps):
    """
    X: array of shape (N, F)
    y: array of shape (N,) or (N,1)
    returns
      X_seq: (N - time_steps, time_steps, F)
      y_seq: (N - time_steps,   1)   # aligned so y_seq[i] is the label for X_seq[i]
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps])
    return np.stack(Xs), np.stack(ys)

def get_tf_callbacks():
    earlystop_cb = EarlyStopping(
        monitor='val_loss',        # metric to watch
        min_delta=1e-4,            # minimum change to qualify as improvement
        patience=5,                # how many epochs with no improvement before stopping
        verbose=1,                 # prints a message when stopping
        restore_best_weights=True  # at end of training, rolls back to the best weights
    )
    reducelr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,      # reduce LR by this factor
        patience=3,      # after how many bad epochs
        verbose=1
    )
    return earlystop_cb, reducelr_cb

def run_forecasts(model_builder, params, tickers, paths):
    forecasts = {}
    train_rmse_list = []
    test_rmse_list = []
    for ticker in tickers:
        print('Training and Forecasting for:', ticker)
        target = 'Returns'
        forecast_horizon = 11
        df = data.get_ticker_df(ticker, paths['test_path'], paths['train_stocks_path'])
        df = data.add_indice_features(df, target=target, train_indices_path=paths['train_indices_path'])
        df = data.prepare_features(df, target=target, beta_window=10, ma_windows=[10, 20, 60], ewm_alpha=[0.1, 0.3, 0.5], lags=1, out_len=500)
        train_df, test_df = data.split_df(df, train_ratio=0.8)
        X_train, y_train = get_X_y_multistep(train_df, steps=forecast_horizon, target='Returns')
        X_test, y_test = get_X_y_multistep(test_df, steps=forecast_horizon, target='Returns')

        estimator = model_builder(params)
        if type(estimator) not in [KerasFunctional, KerasSequential]:
            estimator.fit(X_train, y_train)
            X_train_scaled = X_train
            X_test_scaled = X_test
            y_hat_train = pd.DataFrame(estimator.predict(X_train_scaled), index=y_train.index, columns=y_train.columns)
            y_hat_test = pd.DataFrame(estimator.predict(X_test_scaled), index=y_test.index, columns=y_test.columns)
            train_rmse = root_mean_squared_error(y_train, y_hat_train)
            test_rmse = root_mean_squared_error(y_test, y_hat_test)
        else:
            Xscaler, yscaler, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = build_scalers(X_train, y_train, X_test, y_test)
            X_train_scaled_seq, y_train_scaled_seq = make_sequences(X_train_scaled, y_train_scaled, params['window_size'])
            X_test_scaled_seq,  y_test_scaled_seq  = make_sequences(X_test_scaled, y_test_scaled, params['window_size'])
            earlystop_cb, reducelr_cb = get_tf_callbacks()
            estimator.fit(
                X_train_scaled_seq,
                y_train_scaled_seq,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_scaled_seq, y_test_scaled_seq),
                verbose=1,
                callbacks=[earlystop_cb, reducelr_cb] # , tensorboard_cb
                )
            # train performance
            y_hat_train_scaled = estimator.predict(X_train_scaled_seq)
            y_hat_train = yscaler.inverse_transform(y_hat_train_scaled)
            y_hat_train = pd.DataFrame(y_hat_train, columns=y_train.columns, index=y_train.index[params['window_size']:])
            # test prediction
            y_hat_test_scaled = estimator.predict(X_test_scaled_seq)
            y_hat_test= yscaler.inverse_transform(y_hat_test_scaled)
            y_hat_test = pd.DataFrame(y_hat_test, columns=y_test.columns, index=y_test.index[params['window_size']:])
            train_rmse = root_mean_squared_error(y_train[params['window_size']:], y_hat_train)
            test_rmse = root_mean_squared_error(y_test[params['window_size']:], y_hat_test)
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        print((f"Train RMSE: {train_rmse:.5f}\n" f"Test RMSE: {test_rmse:.5f}\n"))
        if type(estimator) not in [KerasFunctional, KerasSequential]:
            forecast = make_forecast(df, estimator, target, columns=y_test.columns)
        else:
            forecast = make_scaled_forecast_4sequence(df, estimator, target, columns=y_test.columns, Xscaler=Xscaler, yscaler=yscaler, window_size=params['window_size'])
        forecasts[ticker] = forecast

    avg_train_rmse = sum(train_rmse_list) / len(train_rmse_list)
    avg_test_rmse = sum(test_rmse_list) / len(test_rmse_list)
    print(f'Average Train RMSE: {avg_train_rmse:.5f}, Average Test RMSE: {avg_test_rmse:.5f}')
    return forecasts

def create_submission_file(forecasts, rootpath, test_files, fnamesuffix=''):
    fpath = os.path.join(rootpath, 'data', 'stock-price-prediction-challenge', 'sample_submission.csv')
    
    sample_submission = pd.read_csv(fpath)
    # sample_submission
    dates_from_sample = sample_submission['Date']

    submission_df = pd.DataFrame(dates_from_sample, columns=['Date'])

    for test_file in test_files:
        predictions = forecasts[test_file].iloc[:, -10:].values.reshape(-1, 1)
        test_no = test_file.split('_')[1]
        col = f'Returns_{test_no}'
        submission_df[col] = predictions
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fpath = f'Regression_submission{fnamesuffix}_{now}.csv'
    submission_df.to_csv(fpath, index=False)
    print(f'Submission file created: {fpath}')