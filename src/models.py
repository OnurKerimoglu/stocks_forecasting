from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper
from skopt.space import Real
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class VStackedEstimatorMultiTarget:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None
        self.stack_cols = None

    def fit(self, X_1, X_2, y, stack_cols=None):
        # Train model_1
        self.model_1.fit(X_1, y)

        # Make predictions
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index,
            columns=y.columns,
        )
        # Compute residuals
        y_resid = y - y_fit
        y_resid = y_resid.stack(stack_cols).squeeze()  # wide to long

        # Train model_2 on residuals
        self.model_2.fit(X_2, y_resid)

        # Save column names for predict method
        self.y_columns = y.columns
        self.stack_cols = stack_cols

    def predict(self, X_1, X_2):
        # Predict with model_1
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index,
            columns=self.y_columns,
        )
        y_pred = y_pred.stack(self.stack_cols).squeeze()  # wide to long

        # Add model_2 predictions to model_1 predictions
        y_pred += self.model_2.predict(X_2)
        return y_pred.unstack(self.stack_cols)


class VStackedEstimatorSingleTarget:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2

    def fit(self, X_1, X_2, y):
        # Train model_1
        self.model_1.fit(X_1, y)
        # Make predictions with model_1
        y_pred1 = self.model_1.predict(X_1)
        # Compute residuals
        y_resid = y - y_pred1
        # Train model_2 on residuals
        self.model_2.fit(X_2, y_resid)

    def predict(self, X_1, X_2):
        # Predict with model_1
        y_pred = self.model_1.predict(X_1)
        # Add model_2 predictions to model_1 predictions
        y_pred += self.model_2.predict(X_2)
        # y_pred.unstack(self.stack_cols)
        return y_pred

def evaluate(y_train, y_hat_train, y_test, y_hat_test):
    rmse_train = root_mean_squared_error(y_train, y_hat_train)
    print(f'Training RMSE: {rmse_train:.5f}')
    rmse_test = root_mean_squared_error(y_test, y_hat_test)
    print(f'Test RMSE: {rmse_test:.5f}')

    y_train.plot(style=".", color='blue')
    y_test.plot(style=".", color='red')
    y_hat_train.plot(style='-', color='blue')
    y_hat_test.plot(style='-', color='red')

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

def evaluate_multistep_forecast(y_train, y_train_hat, y_test, y_test_hat, df, ticker, target):
    # Used later for annotation
    train_rmse = root_mean_squared_error(y_train, y_train_hat)
    test_rmse = root_mean_squared_error(y_test, y_test_hat)
    print((f"{ticker}: Train RMSE: {train_rmse:.5f}, Test RMSE: {test_rmse:.5f}"))

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
    ax1 = df[target][y_train_hat.index].plot(**plot_params, ax=ax1)
    ax1 = plot_multistep(y_train_hat, ax=ax1, palette_kwargs=palette)
    ax1.set(xlabel='')
    _ = ax1.legend([f'{target} (train)', 'Forecast'], loc="upper right")
    ax2 = df[target][y_test_hat.index].plot(**plot_params, ax=ax2)
    ax2 = plot_multistep(y_test_hat, ax=ax2, palette_kwargs=palette)
    ax2.set(xlabel='')
    _ = ax2.legend([f'{target} (test)', 'Forecast'], loc="upper right")
    plt.suptitle(f'{ticker} Train RMSE: {train_rmse:.5f}, Test RMSE: {test_rmse:.5f}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def optimize_elasticnet_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features2scale: list,
    cv_splits: int = 4,
    n_iter: int = 30
):
    """
    Builds a Pipeline with preprocessing and ElasticNet, then runs BayesSearchCV.

    Parameters
    ----------
    X_train : DataFrame of training features
    y_train : Series of training targets
    features2scale : list of column names to StandardScale
    cv_splits : number of TimeSeriesSplit folds
    n_iter : BayesSearchCV iterations

    Returns
    -------
    best_pipeline : Pipeline
        Pipeline refit on full training data with best hyperparameters
    best_params : dict
        Best-found hyperparameters
    """
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features2scale),
        ],
        remainder='passthrough'
    )

    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', ElasticNet(max_iter=5000, random_state=42))
    ])

    # Hyperparameter search space
    search_spaces = {
        'model__alpha': Real(1e-3, 10.0, prior='log-uniform'),  # Constant that multiplies the penalty terms. alpha=0.0 is ordinary linear regression
        'model__l1_ratio': Real(0.0, 1.0, prior='uniform')  # The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. 
    }

    # Time series CV
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # stop if best objective hasn't improved in a while
    stopper = DeltaYStopper(delta=1e-2, n_best=5)

    # BayesSearchCV
    bayes_cv = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    # Fit
    bayes_cv.fit(X_train, y_train, callback=stopper)
    print(f"Best hyperparameters: {bayes_cv.best_params_}")

    return bayes_cv.best_estimator_, bayes_cv.best_params_