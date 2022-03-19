from models.ARIMA import arima_forecast
from datetime import datetime, timedelta
import pytz

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.token import TOKEN
import pandas as pd
import time
from dateutil import parser
from os import listdir
from os.path import isfile, join, splitext

from models.ARIMA.arima_forecast import ArimaPredictor
from .dataLoaderService import get_current_price


def get_arima_prediction(pkl_name: str):
    last_data_elem = parser.parse(splitext(pkl_name)[0].split("#", 2)[1])
    predictor = ArimaPredictor(last_data_elem)
    predictor.load_model(pkl_name)

    return predictor.predict(1, 'H'), \
           predictor.predict(12, 'H'), \
           predictor.predict(24, 'H'), \
           predictor.predict(48, 'H')


def find_pkl_name(ticker, models_url):
    all_models = [f for f in listdir(models_url) if isfile(join(models_url, f))]
    for model_filename in all_models:
        if model_filename.startswith(ticker):
            return f"{models_url}/{model_filename}"
    return None


def get_prediction(ticker: str, figi: str, data_url: str):
    pkl_name = find_pkl_name(ticker, data_url)
    if pkl_name is None:
        print(f"Trained model is not found for ticker:={ticker}")
        return None

    current_price = get_current_price(ticker, figi)
    return current_price, *get_arima_prediction(pkl_name)
