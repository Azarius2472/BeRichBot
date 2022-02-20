import json


def apiRichBotGetAvailableCompanies():
    # TODO rewrite when API is ready
    with open("mocks/companies.mock.json") as json_file:
        data = json.load(json_file)
    return data

def apiRichBotGetCompanyByTicker(ticker):
    # TODO rewrite when API is ready
    with open("mocks/companybyticker.mock.json") as json_file:
        data = json.load(json_file)
    resultText = '\n' + data[0]['name'] + '\n ABOUT: ' + '\n' + data[0]['about']

    return resultText

def apiRichBotGetPredictionForCompany(ticker):
    # TODO rewrite when API is ready
    with open("mocks/prediction.mock.json") as json_file:
        data = json.load(json_file)
    return data
