import pandas as pd
from sklearn.metrics import root_mean_squared_error


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