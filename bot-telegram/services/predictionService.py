from models.ARIMA import arima_forecast
from datetime import datetime, timedelta

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.token import TOKEN
import pandas as pd
import time


def get_prediction(ticker: str, figi: str, data_url: str):
    csv_name = f"{data_url}/{ticker}.csv"
    data = pd.read_csv(csv_name, index_col='time', parse_dates=['time'])

    last_time = data['time'].iloc[-1]
    now = datetime.now() + timedelta(hours=3)

    data_update = build_dataframe_for_figi(figi, last_time, now)


def build_dataframe_for_figi(company, _from, to):
    start_time = time.time()
    candles = []
    with Client(TOKEN) as client:
        for i in range(step, days + 1, step):
            from_ = now - timedelta(days=i)
            to = now - timedelta(days=i - step)
            print(f"Iteration:={i // step} from:={from_} to:= {to}")

            error = {}
            while error is not None:
                try:
                    response = list(client.get_all_candles(figi=company['figi'], from_=from_, to=to,
                                                           interval=CandleInterval.CANDLE_INTERVAL_1_MIN))
                    candles.extend(response)
                    error = None
                except Exception as e:
                    error = e
                    print(f"Request limit. Waiting 60 sec...")
                    time.sleep(60)

    print(f"Loaded data for company :={company['ticker']} size:={len(candles)}  time:={time.time() - start_time}")

    candles_as_dict = []
    for c in candles:
        c_open = float(str(c.open.units) + "." + str(c.open.nano))
        c_close = float(str(c.close.units) + "." + str(c.close.nano))
        c_high = float(str(c.high.units) + "." + str(c.high.nano))
        c_low = float(str(c.low.units) + "." + str(c.low.nano))

        c_dict = c.__dict__
        c_dict['open'] = c_open
        c_dict['close'] = c_close
        c_dict['high'] = c_high
        c_dict['low'] = c_low
        del c_dict['is_complete']
        candles_as_dict.append(c_dict)

    return pd.DataFrame(candles_as_dict)
