from datetime import datetime
import pytz
from .dataLoaderService import build_dataframe_for_figi
import pandas as pd

from os import listdir
from os.path import isfile, join, splitext


def updateDataToCurrent(ticker: str, figi: str, data_url: str):
    csv_name = f"{data_url}/{ticker}.csv"
    data = pd.read_csv(csv_name, parse_dates=['time'])

    last_time = pd.to_datetime(data['time']).iloc[-1]
    now = datetime.now(pytz.utc)

    data_update = build_dataframe_for_figi(ticker, figi, last_time, now)
    enriched_df = pd.concat([data, data_update])
    enriched_df = enriched_df.drop_duplicates(subset=['time'])
    enriched_df = enriched_df.set_index('time')
    enriched_df.to_csv(csv_name)
    return enriched_df


if __name__ == '__main__':
    companies_file_name = "../../data/companies_dataset.csv"
    companies = pd.read_csv(companies_file_name)

    mypath = "D:/2022/reduced-trading-ds"
    onlyfiles = [splitext(f)[0] for f in listdir(mypath) if isfile(join(mypath, f))]
    reduced = companies[companies.ticker.isin(onlyfiles)]

    reduced_csv_name = "../../data/reduced_dataset.csv"
    reduced.to_csv(reduced_csv_name)
