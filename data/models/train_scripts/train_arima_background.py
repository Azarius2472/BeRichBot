import time

import pandas as pd

from os import listdir
from os.path import splitext

from models.ARIMA.arima_forecast import ArimaPredictor
from bot_telegram.services.modelsMaintenanceService import updateDataToCurrent
from bot_telegram.services.companyService import getByTickerOrName
from bot_telegram.consts.commonConst import DATA_PATH

dataset_path = "D:/2022/reduced-trading-ds"
arima_model_path = "D:/2022/be-rich-bot/data/models/arima"

if __name__ == '__main__':

    while True:
        for f in listdir(dataset_path):
            ticker = splitext(f)[0]
            print(f"{ticker}\n")

            company = getByTickerOrName(ticker)[0]
            data = updateDataToCurrent(ticker, company['figi'], DATA_PATH)

            p = ArimaPredictor()

            p.train(data, freq='H')

            date_path_part_str = str(p.last_data_elem).replace(':', '_')
            pkl_name = f"{arima_model_path}/{ticker}#{date_path_part_str}.pkl"
            p.save_model(pkl_name)

        time.sleep(60 * 120)
