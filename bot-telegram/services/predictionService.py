from datetime import datetime

from dateutil import parser
from os import listdir
from os.path import isfile, join, splitext

from models.ARIMA.arima_forecast import ArimaPredictor
from models.dl_predictor import TSPredictor
from .dataLoaderService import get_current_price
import pandas as pd
from .dataLoaderService import get_last_hour_data
import numpy as np


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


def find_arima_pkl_name(ticker, models_url):
    arima_models_url = f"{models_url}/arima"
    all_models = [f for f in listdir(arima_models_url) if isfile(join(arima_models_url, f))]
    for model_filename in all_models:
        if model_filename.startswith(ticker):
            return f"{arima_models_url}/{model_filename}"
    return None


def get_lstm_prediction(ticker, data_url, models_url):
    pass


def get_prediction(ticker: str, figi: str, data_url: str, models_url: str):
    pkl_name = find_arima_pkl_name(ticker, models_url)
    if pkl_name is None:
        print(f"Trained model is not found for ticker:={ticker}")
        return None

    current_price = get_current_price(ticker, figi)

    csv_name = f"{data_url}/{ticker}.csv"
    data = get_last_hour_data(figi)

    cnn_prediction = get_dl_prediction(ticker, data, models_url, 'CNN')
    lstm_prediction = get_dl_prediction(ticker, data, models_url, 'LSTM')
    arima_prediction = get_arima_prediction(pkl_name)

    final_predictions = np.mean(np.array([cnn_prediction, lstm_prediction, arima_prediction]), axis=0).tolist()
    return current_price, *final_predictions


def get_dl_prediction(ticker, data, models_url, model_type):
    path_to_model = f"{models_url}/{model_type.lower()}/{ticker}_{model_type}.pt"
    predictor = TSPredictor(needLoad=True, model_type=model_type, path_to_model=path_to_model)

    data_for_predictions = data[-60:]
    predicted_values = predictor.inference_far_period(data_for_predictions, 48 * 60, 'cpu')

    return predicted_values[-48], \
           predicted_values[-36], \
           predicted_values[-24], \
           predicted_values[-1]
