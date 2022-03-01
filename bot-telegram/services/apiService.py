import json
from .companyService import getAll
from .companyService import getByTickerOrName


def apiRichBotGetAvailableCompanies():
    # TODO add pagination
    return getAll()


def apiRichBotGetCompanyByTickerOrName(ticker):
    data = getByTickerOrName(ticker)
    if len(data) == 0:
        return None

    resultText = '\n\n' + data[0]['name'] + '\nTicker: ' + '\t' + data[0]['ticker']
    return resultText


def apiRichBotGetPredictionForCompany(ticker):
    # TODO rewrite when API is ready
    with open("mocks/prediction.mock.json") as json_file:
        data = json.load(json_file)
    response = reformatPrediction(data)
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
