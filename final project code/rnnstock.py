import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN,LSTM
from keras.layers import Dropout
dataset_train = pd.read_csv('train.csv')
dataset_train = dataset_train.sort_values(by='Date').reset_index(drop=True)

training_set = dataset_train.iloc[:, 4:5].values


sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []
for i in range(60, 2416):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape)
print(y_train.shape)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)
regressor = Sequential()


regressor.add(SimpleRNN(units = 50, input_shape = (X_train.shape[1], 1)))

regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split=0.1)

regressor.summary()
dataset_test = pd.read_csv('val.csv')
dataset_test = dataset_test.sort_values(by='Date').reset_index(drop=True)

real_stock_price = dataset_test.iloc[:, 4:5].values

dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = []
for i in range(60, 866):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
MSE = sum(pow((predicted_stock_price - real_stock_price),2))/predicted_stock_price.shape[0]
RMSE = np.sqrt(MSE)

print('MSE',sum(pow((predicted_stock_price - real_stock_price),2))/predicted_stock_price.shape[0])
print('MAE',sum(abs(predicted_stock_price - real_stock_price))/predicted_stock_price.shape[0])


plt.plot(real_stock_price, color = 'black', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TAT Stock Price')
plt.title('RNN Stock Price Prediction')
plt.xlabel('samples')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()
def high_low_breakout_backtest(real_stock_price, window_size, predicted_stock_price, initial_cash=500000):
    """
    High-Low Breakout strategy backtest function with predicted prices.

    :param real_stock_price: The price series of the asset.
    :param window_size: The window size for calculating the highest high and lowest low.
    :param predicted_stock_price: The predicted price series of the asset.
    :param initial_cash: The initial cash in the portfolio.
    :return: The final portfolio value and the total return.
    """
    cash = initial_cash
    shares = 0
    total_assets = initial_cash
    real_stock_price = pd.Series(real_stock_price)
    predicted_stock_price = pd.Series(predicted_stock_price)

    highest_high = real_stock_price.rolling(window_size).max()
    lowest_low = real_stock_price.rolling(window_size).min()

    positions = np.where(predicted_stock_price > highest_high, -1, np.where(predicted_stock_price < lowest_low, 1, 0))  # Generate trading signals

    for i in range(1, len(real_stock_price)):
        if positions[i] > positions[i - 1] and cash >= real_stock_price[i]:
            shares_bought = cash // real_stock_price[i]
            shares += shares_bought
            cash -= shares_bought * real_stock_price[i]
        elif positions[i] < positions[i - 1] and shares > 0:
            cash += shares * real_stock_price[i]
            shares = 0

        total_assets = cash + shares * real_stock_price[i]

    total_return = total_assets - initial_cash
    return total_assets, total_return
initial_cash = 500000
# Applying the high low breakout backtest function to the LSTM model's predictions
final_value, total_return = high_low_breakout_backtest(real_stock_price[:, 0], 20, predicted_stock_price[:,0])
return_rate = (total_return/initial_cash)*100
print("High_Low_Final Assets:", final_value, "High_Low_Total Return:", total_return, "High_Low_Return Rate:",return_rate)


#R-Breaker Backtest
def r_breaker_backtest(real_stock_price, predicted_stock_price, window_size, k, initial_cash=500000):
    """
    R-Breaker strategy backtest function.

    :param real_stock_price: The actual price series of the asset.
    :param predicted_stock_price: The predicted price series of the asset.
    :param window_size: The window size for calculating the highest high and lowest low.
    :param k: The parameter for calculating the breakout thresholds.
    :param initial_cash: The initial cash in the portfolio.
    :return: The final portfolio value and the total return.
    """
    cash = initial_cash
    shares = 0
    total_assets = initial_cash

    for i in range(window_size, len(real_stock_price)):
        highest_high = real_stock_price[i-window_size:i].max()
        lowest_low = real_stock_price[i-window_size:i].min()
        range_high = real_stock_price[i-1]
        range_low = real_stock_price[i-1]
        range_break = k * (range_high - range_low)

        sell_zone_upper = range_high + range_break
        buy_zone_lower = range_low - range_break
        reverse_sell_zone_upper = range_high + 0.5 * range_break
        reverse_buy_zone_lower = range_low - 0.5 * range_break
        breakout_buy_zone_upper = range_high + 2 * range_break
        breakout_sell_zone_lower = range_low - 2 * range_break

        position = 0
        if predicted_stock_price[i] > sell_zone_upper:
            position = -1
        elif predicted_stock_price[i] < buy_zone_lower:
            position = 1
        elif predicted_stock_price[i] > reverse_sell_zone_upper and predicted_stock_price[i] <= sell_zone_upper:
            position = -0.5
        elif predicted_stock_price[i] < reverse_buy_zone_lower and predicted_stock_price[i] >= buy_zone_lower:
            position = 0.5
        elif predicted_stock_price[i] > breakout_buy_zone_upper:
            position = 1
        elif predicted_stock_prices[i] < breakout_sell_zone_lower:
            position = -1

        if position > 0 and cash >= real_stock_price[i]:
            shares_bought = cash // real_stock_price[i]
            shares += shares_bought
            cash -= shares_bought * real_stock_price[i]
        elif position < 0 and shares > 0:
            cash += shares * real_stock_price[i]
            shares = 0

        total_assets = cash + shares * real_stock_price[i]

    total_return = total_assets - initial_cash
    return total_assets, total_return
initial_cash = 500000
# Applying the R-breaker breakout backtest function to the LSTM model's predictions
final_value1, total_return = r_breaker_backtest(real_stock_price[:, 0], predicted_stock_price[:, 0], 20, 0.5)
return_rate = (total_return/initial_cash)*100
print("R_Breaker_Final Assets:", final_value1, "R_Breaker_Total Return:", total_return, "R_Breaker_Return Rate:",return_rate)