import telebot

from services.mainService import getGreetingsMenu
from services.mainService import getCommandsMenu
from services.mainService import getCompanyMenu
from services.mainService import getCompanyListMenu

from consts.textConsts import CONTACTS_TEXT
from consts.textConsts import INFO_ABOUT_BOT_TEXT
from consts.tokenConst import BOT_TOKEN
from consts.commonConst import DATA_PATH, MODELS_PATH

from services.apiService import apiRichBotGetAvailableCompanies
from services.apiService import apiRichBotGetCompanyByTickerOrName
from services.apiService import apiRichBotGetPredictionForCompany

# Создаем экземпляр бота
bot = telebot.TeleBot(BOT_TOKEN)

# States
FIRST, SECOND = range(2)

# Callback data
GET_COMPANY = 'getCompany'
GET_ALL_COMPANIES = 'getAllCompanies'
GET_INFO = 'getInfo'
GET_CONTACTS = 'getContacts'
PREDICTION = 'prediction'

users = {}


class CompaniesStore:

    def __init__(self, companies):
        self.companies = companies

    def getCompanies(self):
        return self.companies

    def setCompanies(self, companies):
        self.companies = companies


class States:
    START = 0
    CHOOSE = 1
    COMPANIES = 2


# Функция, обрабатывающая команду /start
@bot.message_handler(commands=["start"])
def start(message):
    users[str(message.chat.id)] = States.START
    markup = getGreetingsMenu()
    bot.send_message(message.chat.id, message.chat.first_name + ', please, select', reply_markup=markup)


# Получение сообщений от юзера
@bot.message_handler(content_types=["text"])
def handle_text(message):
    if users.get(str(message.chat.id)) is None:
        users[str(message.chat.id)] = States.START

    if users.get(str(message.chat.id)) == States.START:
        markup = getGreetingsMenu()
        markupDefault = getCommandsMenu()
        bot.send_message(message.chat.id, message.chat.first_name + ', please, select',
                         reply_markup=markup)
        bot.send_message(message.chat.id, '^main menu^', reply_markup=markupDefault)

    if users.get(str(message.chat.id)) == States.CHOOSE:
        infoAboutCompany = apiRichBotGetCompanyByTickerOrName(message.text)
        infoAboutCompanyHandler(infoAboutCompany, message)

    if users.get(str(message.chat.id)) == States.COMPANIES:
        infoAboutCompany = apiRichBotGetCompanyByTickerOrName(message.text)
        infoAboutCompanyHandler(infoAboutCompany, message)

    print('message.chat.id', message.chat.id, users)


# Handlers
@bot.callback_query_handler(func=lambda call: call.data == GET_COMPANY)
def getCompany(call):
    bot.send_message(call.from_user.id, call.from_user.first_name + ' please, write Ticker:')
    users[str(call.from_user.id)] = States.CHOOSE


@bot.callback_query_handler(func=lambda call: call.data == GET_ALL_COMPANIES)
def getAllCompanies(call):
    companies = apiRichBotGetAvailableCompanies()
    markup = getCompanyListMenu(companies)
    bot.send_message(call.from_user.id, call.from_user.first_name + ', please, make your choice', reply_markup=markup)
    users[str(call.from_user.id)] = States.COMPANIES


@bot.callback_query_handler(func=lambda call: call.data == GET_INFO)
def getInfo(call):
    markup = getCommandsMenu()
    bot.send_message(call.from_user.id, call.from_user.first_name + ', ' + INFO_ABOUT_BOT_TEXT, reply_markup=markup)


@bot.callback_query_handler(func=lambda call: PREDICTION in call.data)
def getInfo(call):
    companyName = call.data
    companyName = companyName.replace(companyName.split('.')[0] + ".", '')
    predictionText = apiRichBotGetPredictionForCompany(companyName, DATA_PATH, MODELS_PATH)
    markup = getCommandsMenu()
    bot.send_message(call.from_user.id, call.from_user.first_name + ', ' + predictionText,
                     reply_markup=markup,
                     parse_mode="HTML")


@bot.callback_query_handler(func=lambda call: call.data == GET_CONTACTS)
def getContacts(call):
    markup = getCommandsMenu()
    bot.send_message(call.from_user.id, call.from_user.first_name + ', ' + CONTACTS_TEXT, reply_markup=markup)


def errorHandler(message, text):
    markupDefault = getCommandsMenu()
    bot.send_message(message.chat.id, text, reply_markup=markupDefault)


def infoAboutCompanyHandler(infoAboutCompany, message):
    if infoAboutCompany is not None:
        markup = getCompanyMenu(message.text)
        bot.send_message(message.chat.id, message.chat.first_name + ', info about company: ' + infoAboutCompany,
                         reply_markup=markup)
    else:
        errorHandler(message, 'Company Not Found')

    users[str(message.chat.id)] = States.START


# Запускаем бота
bot.polling(none_stop=True, interval=0)
