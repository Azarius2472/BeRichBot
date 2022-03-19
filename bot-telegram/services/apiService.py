import json
from .companyService import getAll
from .companyService import getByTickerOrName
from .predictionService import get_arima_prediction


def apiRichBotGetAvailableCompanies():
    # TODO add pagination
    return getAll()


def apiRichBotGetCompanyByTickerOrName(ticker):
    data = getByTickerOrName(ticker)
    if len(data) == 0:
        return None

    resultText = '\n\n' + data[0]['name'] + '\nTicker: ' + '\t' + data[0]['ticker'] + '\nFigi: ' + '\t' + data[0][
        'figi']
    return resultText


def apiRichBotGetPredictionForCompany(tickerOrName, data_url: str):
    company = getByTickerOrName(tickerOrName)

    if len(company) == 0:
        return None

    company = company[0]
    currentPrice, price5minutes, price1hour, price1day = get_arima_prediction(company['ticker'], company['figi'],
                                                                              data_url)
    predict_result = {
        "name": company['name'],
        "currentPrice": {
            "text": "Current:",
            "value": currentPrice,
            "style": "\\U0001F537"
        },
        "price5minutes": {
            "text": "5 minutes Prediction:",
            "value": price5minutes,
            "style": "\\U0001F53B"
        },
        "price1hour": {
            "text": "1 hour Prediction:",
            "value": price1hour,
            "style": "\\U0001F53B"
        },
        "price1day": {
            "text": "1 day Prediction:",
            "value": price1day,
            "style": "\\U0001F53C"
        }
    }
    response = reformatPrediction(predict_result)
    return response


def reformatPrediction(data):
    text = data[0]
    textString = text['name'] + '\n'
    textString = textString + text['currentPrice']['text'] + '\n' + text['currentPrice']['value'] + ' ' + \
                 text['currentPrice']['style'] + '\n' + '\n'
    textString = textString + text['price5minutes']['text'] + '\n' + text['price5minutes'][
        'value'] + ' ' + text['price5minutes']['style'] + '\n' + '\n'
    textString = textString + text['price1hour']['text'] + '\n' + text['price1hour']['value'] + ' ' + \
                 text['price1hour']['style'] + '\n' + '\n'
    textString = textString + text['price1day']['text'] + '\n' + text['price1day']['value'] + ' ' + text['price1day'][
        'style'] + '\n' + '\n'
    decodedTextString = bytes(textString, "utf-8").decode("unicode_escape")
    return decodedTextString
