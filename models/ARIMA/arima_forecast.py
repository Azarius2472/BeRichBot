import pmdarima as pm
import numpy as np
import pandas as pd
import pickle

def read_data(file, col_name='close'):
    dataset = pd.read_csv(file, index_col='time', parse_dates=['time'])
    data = dataset.loc[:, col_name]
    data = data.to_frame()
    data = data.sort_index()
    return data


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
                           stepwise=True)
    return model


def forecast_arima(file, n_periods, freq='D', use_saved_model=True):
    """
    Create a forecast for time series using ARIMA model.

    Parameters
    ----------
    file: str
    Path to the .csv data

    n_periods: int
    Period for which the forecast is made

    freq: str, default: 'D'
    Frequency of forecast
    'D' - days, 'H' - hours, 'T' - 'minutes'
    Example: n_periods=5, freq='D' -> forecast for 5 days

    use_saved_model: bool, default: True
    If we use already saved model to predict

    Returns
    -------
    List of predicted values
    """

    data = read_data(file)
    if freq == 'D':
        data = data.resample('D').mean()
        data = data.fillna(data.bfill())

    if use_saved_model:
        with open('auto_arima.pkl', 'rb') as pkl:
            model = pickle.load(pkl)

        model.fit(data)  # update the model with the new data
    else:
        model = train_arima(data)

    fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(start=data.index[-1], periods=n_periods, freq=freq)

    fitted_series = pd.DataFrame(fitted, index=index_of_fc, columns=['Value'])
    return list(fitted_series.Value)

