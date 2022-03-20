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
    last_data_str = splitext(pkl_name)[0].split("#", 2)[1]
    last_data_str = last_data_str.replace('_', ':')
    last_data = parser.parse(last_data_str)

    predictor = ArimaPredictor(last_data)
    predictor.load_model(pkl_name)

    time_delta = datetime.utcnow().replace(tzinfo=None) - last_data.replace(tzinfo=None)
    missed_hours = time_delta.days * 24 + time_delta.seconds // 3600

    print(f"Missed hours from {last_data} = {missed_hours}")
    predicted_values = predictor.predict(missed_hours + 48, 'H')
    return predicted_values[-48], \
           predicted_values[-36], \
           predicted_values[-24], \
           predicted_values[-1]


def find_pkl_name(ticker, models_url):
    all_models = [f for f in listdir(models_url) if isfile(join(models_url, f))]
    for model_filename in all_models:
        if model_filename.startswith(ticker):
            return f"{models_url}/{model_filename}"
    return None


def get_prediction(ticker: str, figi: str, models_url: str):
    pkl_name = find_pkl_name(ticker, models_url)
    if pkl_name is None:
        print(f"Trained model is not found for ticker:={ticker}")
        return None

    current_price = get_current_price(ticker, figi)
    return current_price, *get_arima_prediction(pkl_name)
