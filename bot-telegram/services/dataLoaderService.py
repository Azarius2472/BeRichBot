from tinkoff.invest import CandleInterval, Client

try:
    from tinkoff.invest.token import TOKEN, InvestTokenNotFound
except Exception as e:
    print(e)
    TOKEN = None

import pandas as pd
import time
from datetime import datetime, timedelta


def build_dataframe_for_figi(ticker: str, figi: str, from_, to):
    start_time = time.time()
    candles = []
    with Client(TOKEN) as client:
        print(from_, to)
        error = {}
        while error is not None:
            try:
                response = list(client.get_all_candles(figi=figi, from_=from_, to=to,
                                                       interval=CandleInterval.CANDLE_INTERVAL_1_MIN))
                candles.extend(response)
                error = None
            except Exception as e:
                error = e
                print(e)
                print(f"Request limit. Waiting 60 sec...")
                time.sleep(60)

    print(f"Loaded data for company :={ticker} size:={len(candles)}  time:={time.time() - start_time}")

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


def get_current_price(ticker: str, figi: str):
    candles = []

    if TOKEN is None:
        return -1

    with Client(TOKEN) as client:
        error = {}
        from_data = datetime.utcnow() - timedelta(days=1)
        while error is not None:
            try:
                response = list(client.get_all_candles(figi=figi, from_=from_data,
                                                       interval=CandleInterval.CANDLE_INTERVAL_HOUR))

                if len(response) == 0:
                    from_data = from_data - timedelta(hours=2)
                    continue

                candles.extend(response)
                error = None
            except Exception as e:
                from_data = from_data - timedelta(hours=2)
                error = e
                print(e)

    return float(str(candles[-1].close.units) + "." + str(candles[-1].close.nano))
