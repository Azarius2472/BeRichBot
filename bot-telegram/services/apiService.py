import json
from .companyService import getAll
from .companyService import getByTickerOrName
from .predictionService import get_prediction


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


def apiRichBotGetPredictionForCompany(tickerOrName, data_url: str, models_url: str):
    company = getByTickerOrName(tickerOrName)

    if len(company) == 0:
        return None

    company = company[0]
    currentPrice, price1hors, price12hors, price24hors, price48hors = get_prediction(company['ticker'], company['figi'],
                                                                                     data_url,
                                                                                     models_url)
    predict_result = {
        "name": company['name'],
        "currentPrice": {
            "text": "Current:",
            "value": str(currentPrice),
            "style": "\\U0001F537"
        },
        "price1hors": {
            "text": "1 hour Prediction:",
            "value": str(price1hors),
            "style": getStickerForPrediction(currentPrice, price1hors)
        },
        "price12hors": {
            "text": "12 hours Prediction:",
            "value": str(price12hors),
            "style": getStickerForPrediction(currentPrice, price12hors)
        },
        "price24hors": {
            "text": "24 hours Prediction:",
            "value": str(price24hors),
            "style": getStickerForPrediction(currentPrice, price24hors)
        },
        "price48hors": {
            "text": "48 hours Prediction:",
            "value": str(price48hors),
            "style": getStickerForPrediction(currentPrice, price48hors)
        }
    }
    response = reformatPrediction(predict_result)
    return response


def getStickerForPrediction(currectValue, predictionValue):
    if currectValue > predictionValue:
        return "\\U0001F53B"
    return "\\U0001F53C"


def reformatPrediction(data):
    text = data
    textString = text['name'] + '\n'
    textString = textString + text['currentPrice']['text'] + '\n' + text['currentPrice']['value'] + ' ' + \
                 text['currentPrice']['style'] + '\n' + '\n'

    textString = textString + text['price1hors']['text'] + '\n' + text['price1hors'][
        'value'] + ' ' + text['price1hors']['style'] + '\n' + '\n'

    textString = textString + text['price12hors']['text'] + '\n' + text['price12hors']['value'] + ' ' + \
                 text['price12hors']['style'] + '\n' + '\n'

    textString = textString + text['price24hors']['text'] + '\n' + text['price24hors']['value'] + ' ' + \
                 text['price24hors']['style'] + '\n' + '\n'

    textString = textString + text['price48hors']['text'] + '\n' + text['price48hors']['value'] + ' ' + \
                 text['price48hors']['style'] + '\n' + '\n'

    decodedTextString = bytes(textString, "utf-8").decode("unicode_escape")
    return decodedTextString
