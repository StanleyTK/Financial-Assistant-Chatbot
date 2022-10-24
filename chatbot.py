import random
import json
import pickle
import requests
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

global intents
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Utilizes the nltk library to lemmatize the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Returns the sentence after
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Predicts what the class corresponds to the sentence in the intents list
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.50
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list



# Gets the response from the intents json file using the intents list
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])

    print(result)
    if tag == 'stockgraph':
        stockgraph(False)
    elif tag == 'gainers':
        topGainers()
    elif tag == 'losers':
        topLosers()
    elif tag == 'search':
        searchStockPrice()
    elif tag == 'predict':
        stockgraph(True)
    return result



# Prints out the stats of the top losers
def topLosers():
    response = requests.get(
        "https://financialmodelingprep.com/api/v3/stock_market/losers?apikey=85f949784452240595cc529a315eb1b8")
    while (True):
        index = int(input("how many would you like?"))
        if (index > 30):
            print("Please input an number that is less or equal to 30")
        else:
            break
    print("Here are the top " + str(index) + " losers of the day")
    if response.status_code == 200:
        json_resp = response.json()
        for i in range(index):
            name = json_resp[i]['name']
            symbol = json_resp[i]['symbol']
            percent = json_resp[i]['changesPercentage']
            price = json_resp[i]['price']
            change = json_resp[i]['change']
            print(str(i+1) + ".")
            print("Name: " + name)
            print("Symbol: " + symbol)
            print("Percent Change: " + str(percent))
            print("Current Price: " + str(price))
            print("Change: " + str(change) + "\n")

    else:
        print("Cannot connect to the server, sorry.")
    print("Is there anything else I can do for you?")


# Prints out the stats of the top gainers
def topGainers():
    response = requests.get(
        "https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey=85f949784452240595cc529a315eb1b8")
    while (True):
        index = int(input("how many would you like?"))
        if (index > 30):
            print("Please input an number that is less or equal to 30")
        else:
            break
    print("Here are the top " + str(index) + " winners of the day")
    if response.status_code == 200:
        json_resp = response.json()
        for i in range(index):
            print(str(i + 1) + ".\n")
            print("Name: " + json_resp[i]['name'])
            print("Symbol: " + json_resp[i]['symbol'])
            print("Percent Change: " + str(json_resp[i]['changesPercentage']))
            print("Current Price: " + str(json_resp[i]['price']))
            print("Change:" + str(json_resp[i]['change']) + "\n")
    else:
        print("Cannot connect to the server, sorry.")

    print("Is there anything else I can do for you?")


# Gets the input of the stock, prints out the open, high, percent change, close, volume hourly
# API Key is Outdated. Create another API
def searchStockPrice():
    stockticker = (input("What stock do you want to look up (Stock Ticker)")).upper()
    response = requests.get(
        "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=" + stockticker + "&interval=60min&outputsize=full&apikey=ZGE19H5HOBZKLLHD")
    if response.status_code == 200:
        json_resp = response.json()
        metadata = json_resp["Meta Data"]["3. Last Refreshed"]
        open = json_resp["Time Series (60min)"][metadata]["1. open"]
        high = json_resp["Time Series (60min)"][metadata]["2. high"]
        low = json_resp["Time Series (60min)"][metadata]["3. low"]
        close = json_resp["Time Series (60min)"][metadata]["4. close"]
        volume = json_resp["Time Series (60min)"][metadata]["5. volume"]
        print("\nOpen (Hourly) : " + str(open))
        print("High (Hourly): " + str(high))
        print("Percent Change: " + str(low))
        print("Close: " + str(close))
        print("Volume: \n" + str(volume))
    else:
        print("Cannot connect to the server, sorry.")

    print("Is there anything else I can do for you?")




# STOCKGRAPH, MODEL STILL UNDER CONSTRUCTION
def stockgraph(predict):
    # loading data
    if predict:
        company = input("What company would you like me to predict? (stock tinker)")
    else:
        company = input("What company would you like me to graph? (stock tinker)")

    company = company.upper()
    prediction_days = 60

    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2020, 1, 1)
    data = web.DataReader(company, 'yahoo', start, end)

    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Building the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Testing the Model Accuracy on Existing Data

    # Load Test Data

    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Making Predictions on Test Data
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color="green", label=f"Line of Best Fit")
    plt.title(f"{company} share price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')

    #Predicting the Next Day (INACCURATE!!)
    if predict:
        real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        print(f"Prediction Next Day: {prediction}")
    else:
        plt.legend()
        plt.show()
    print("Is there anything else I can do for you?")
    main_chatbot()




def main_chatbot():
    while True:
        message = input("")
        ints = predict_class(message)
        get_response(ints, intents)



print("Welcome to the chatbot! How can I help you today?")
main_chatbot()
