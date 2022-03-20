import pmdarima as pm
import numpy as np
import pandas as pd
import pickle


def preprocess_data(dataset, col_name='close'):
    data = dataset.loc[:, col_name]
    data = data.to_frame()
    data = data.sort_index()
    return data


class ArimaPredictor:
    def __init__(self, last_data_elem=None):
        self.model = None
        self.last_data_elem = last_data_elem

    def load_model(self, path: str):
        with open(path, 'rb') as pkl:
            self.model = pickle.load(pkl)

    def predict(self, n_periods, freq='D'):
        if self.model is None:
            return None

        fitted, confint = self.model.predict(n_periods=n_periods, return_conf_int=True)
        index_of_fc = pd.date_range(start=self.last_data_elem, periods=n_periods, freq=freq)

        fitted_series = pd.DataFrame(fitted, index=index_of_fc, columns=['Value'])
        return list(fitted_series.Value)

    def train(self, data: pd.DataFrame, freq='D'):
        data = preprocess_data(data)
        if freq == 'D':
            data = data.resample('D').mean()
            data = data.fillna(data.bfill())
        elif freq == 'H':
            data = data.resample('H').mean()
            data = data.fillna(data.bfill())

        if self.model is not None:
            self.model.update(data)  # update the model with the new data
        else:
            self.model = train_arima(data)

        self.last_data_elem = data.index[-1]

    def save_model(self, path: str):
        with open(path, 'wb+') as pkl:
            pickle.dump(self.model, pkl)


def train_arima(data):
    model = pm.auto_arima(data,
                          start_p=1, max_p=3,
                          start_q=1, max_q=3,
                          start_P=0, max_P=5,
                          start_Q=0, max_Q=5,
                          test='adf',  # ADF-test for stationarity detection
                          seasonal=True, m=7,  # seasonality
                          d=1,
                          D=None,
                          trace=True,
                          error_action='warn',
                          suppress_warnings=True,
                          stepwise=True,
                          with_intercept=True, trend='t')
    return model
