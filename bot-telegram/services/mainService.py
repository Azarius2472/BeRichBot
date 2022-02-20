import json

from telebot import types  # для указание типов


def getGreetingsMenu():
    greetKB = types.InlineKeyboardMarkup()
    menuButtons = getGreetingsMenuJSON()

    for buttonValue in menuButtons:
        keyboardButton = types.InlineKeyboardButton(
            text=buttonValue['buttonText'],
            callback_data=buttonValue['buttonFunction'])
        greetKB.add(keyboardButton)

    return greetKB

def getCommandsMenu():
    commandsKB = types.ReplyKeyboardMarkup(resize_keyboard=True)
    menuButtons = getComandsMenuJSON()

    for buttonValue in menuButtons:
        keyboardButton = types.KeyboardButton(text=buttonValue['buttonText'])
        commandsKB.add(keyboardButton)

    return commandsKB

def getCompanyMenu():
    menuKB = types.InlineKeyboardMarkup()
    menuButtons = getCompanyMenuJSON()

    for buttonValue in menuButtons:
        keyboardButton = types.InlineKeyboardButton(text=buttonValue['buttonText'],
                                                    callback_data=buttonValue['buttonFunction'])
        menuKB.add(keyboardButton)

    return menuKB

def getCompanyListMenu(companies):
    commandsKB = types.ReplyKeyboardMarkup(resize_keyboard=True)
    menuButtons = companies

    for buttonValue in menuButtons:
        keyboardButton = types.KeyboardButton(text=buttonValue['name'])
        commandsKB.add(keyboardButton)

    return commandsKB

def getComandsMenuJSON():
    with open("consts/commands.consts.json") as json_file:
        data = json.load(json_file)
    return data

def getGreetingsMenuJSON():
    with open("consts/greetings.consts.json") as json_file:
        data = json.load(json_file)
    return data

def getCompanyMenuJSON():
    with open("consts/company.consts.json") as json_file:
        data = json.load(json_file)
    return data
