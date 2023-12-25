import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
dataset_train = pd.read_csv('train.csv')
training_set = dataset_train.iloc[:, 4:5].values
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []
for i in range(60, 2416):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,y_train,epochs=100,batch_size=32)
dataset_test = pd.read_csv('val.csv')
real_stock_price = dataset_test.iloc[:, 4:5].values
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 866):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print('MSE',sum(pow((predicted_stock_price - real_stock_price),2))/predicted_stock_price.shape[0])
print('MAE',sum(abs(predicted_stock_price - real_stock_price))/predicted_stock_price.shape[0])

plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()
def backtest(predicted_stock_price, real_stock_price, threshold=0.003, initial_cash=500000):
    """
    Backtest function to simulate trading based on the model's predictions.

    :param predictions: The predicted prices from the model.
    :param original_prices: The actual prices of the stock.
    :param threshold: The threshold for making a trade decision.
    :param initial_cash: The initial cash in the portfolio.
    :return: The final portfolio value and the total return.
    """
    cash = initial_cash
    shares = 0
    total_assets = initial_cash

    for i in range(1, len(predicted_stock_price)):
        predicted_change = (predicted_stock_price[i] - real_stock_price[i - 1]) / real_stock_price[i - 1]
        actual_change = (real_stock_price[i] - real_stock_price[i - 1]) / real_stock_price[i - 1]

        # Decision to buy
        if predicted_change > threshold and cash >= real_stock_price[i]:
            shares_bought = cash // real_stock_price[i]
            shares += shares_bought
            cash -= shares_bought * real_stock_price[i]

        # Decision to sell
        elif predicted_change < -threshold and shares > 0:
            cash += shares * real_stock_price[i]
            shares = 0

        # Update total assets value
        total_assets = cash + shares * real_stock_price[i]

    total_return = total_assets - initial_cash
    return total_assets, total_return
initial_cash = 500000
threshold = 0.003
# Applying the backtest function to the LSTM model's predictions
final_assets, total_return = backtest(predicted_stock_price[:, 0], real_stock_price[:, 0], threshold=threshold, initial_cash=initial_cash)
return_rate = (total_return/initial_cash)*100
print("Final Assets:", final_assets, "Total Return:", total_return, "Return Rate:",return_rate)