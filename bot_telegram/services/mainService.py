import json

from telebot import types  # для указание типов


def getGreetingsMenu():
    greetKB = types.InlineKeyboardMarkup()
    menuButtons = getDataFromJsonConst("consts/greetings.consts.json")

    for buttonValue in menuButtons:
        keyboardButton = types.InlineKeyboardButton(
            text=buttonValue['buttonText'],
            callback_data=buttonValue['buttonFunction'])
        greetKB.add(keyboardButton)

    return greetKB


def getCommandsMenu():
    commandsKB = types.ReplyKeyboardMarkup(resize_keyboard=True)
    menuButtons = getDataFromJsonConst("consts/commands.consts.json")

    for buttonValue in menuButtons:
        keyboardButton = types.KeyboardButton(text=buttonValue['buttonText'])
        commandsKB.add(keyboardButton)

    return commandsKB


def getCompanyMenu(company):
    menuKB = types.InlineKeyboardMarkup()
    menuButtons = getDataFromJsonConst("consts/company.consts.json")

    for buttonValue in menuButtons:
        keyboardButton = types.InlineKeyboardButton(text=buttonValue['buttonText'],
                                                    callback_data=buttonValue['buttonFunction']+'.'+company)
        menuKB.add(keyboardButton)

    return menuKB


def getCompanyListMenu(companies):
    commandsKB = types.ReplyKeyboardMarkup(resize_keyboard=True)
    menuButtons = companies

    for buttonValue in menuButtons:
        keyboardButton = types.KeyboardButton(text=buttonValue['name'])
        commandsKB.add(keyboardButton)

    return commandsKB


def getDataFromJsonConst(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data
