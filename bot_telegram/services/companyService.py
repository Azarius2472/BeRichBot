import pandas as pd

companies_file_name = "D:/2022/be-rich-bot/data/reduced_dataset.csv"
companies = pd.read_csv(companies_file_name)


def getAll():
    return companies.to_dict('records')


def getByTickerOrName(ticker_or_name):
    return companies[(companies.ticker == ticker_or_name) | (companies.name == ticker_or_name)].to_dict('records')
