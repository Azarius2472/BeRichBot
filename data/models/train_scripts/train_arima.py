import pandas as pd

from os import listdir
from os.path import splitext

from models.ARIMA.arima_forecast import ArimaPredictor

dataset_path = "D:/2022/reduced-trading-ds"
arima_model_path = "D:/2022/be-rich-bot/data/models/arima"

if __name__ == '__main__':
    for f in listdir(dataset_path):
        ticker = splitext(f)[0]
        print(f"{ticker}\n")
        p = ArimaPredictor()

        csv_name = f"{dataset_path}/{ticker}.csv"
        data = pd.read_csv(csv_name, parse_dates=['time'])
        data = data.set_index('time')

        p.train(data, freq='D')

        date_path_part_str = str(p.last_data_elem).replace(':', '_')
        pkl_name = f"{arima_model_path}/{ticker}#{date_path_part_str}.pkl"
        p.save_model(pkl_name)
