import os

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor

from . import data

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

def run_forecasts(model_builder, params, model_type, tickers, paths):
    forecasts = {}
    train_rmse_list = []
    test_rmse_list = []
    for ticker in tickers:
        print('Training and Forecasting for:', ticker)
        target = 'Returns'
        df = data.get_ticker_df(ticker, paths['test_path'], paths['train_stocks_path'])
        df = data.add_indice_features(df, target=target, train_indices_path=paths['train_indices_path'])
        df = data.prepare_features(df, target=target, beta_window=10, ma_windows=[10, 20, 60], ewm_alpha=[0.1, 0.3, 0.5], lags=1, out_len=500)

        train_df, test_df = data.split_df(df, train_ratio=0.8)
        X_train, y_train = get_X_y_multistep(train_df, steps=11, target='Returns')
        X_test, y_test = get_X_y_multistep(test_df, steps=11, target='Returns')

        estimator = model_builder(params)
        if model_type != 'NN':
            estimator.fit(X_train, y_train)
            X_train_scaled = X_train
            X_test_scaled = X_test
            y_hat_train = pd.DataFrame(estimator.predict(X_train_scaled), index=y_train.index, columns=y_train.columns)
            y_hat_test = pd.DataFrame(estimator.predict(X_test_scaled), index=y_test.index, columns=y_test.columns)
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            estimator.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)
        y_hat_train = pd.DataFrame(estimator.predict(X_train_scaled), index=y_train.index, columns=y_train.columns)
        y_hat_test = pd.DataFrame(estimator.predict(X_test_scaled), index=y_test.index, columns=y_test.columns)
        train_rmse = root_mean_squared_error(y_train, y_hat_train)
        test_rmse = root_mean_squared_error(y_test, y_hat_test)
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        print((f"Train RMSE: {train_rmse:.5f}\n" f"Test RMSE: {test_rmse:.5f}\n"))
        forecast = make_forecast(df, estimator, target, columns=y_test.columns)
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