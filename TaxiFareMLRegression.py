import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.layers import Dense
from Helper import preprocess,feature_engineering

df =pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)
#print(df.head())

# PREPROCESSING DATA
df = preprocess(df)

# FEATURE ENGINEERING
df = feature_engineering(df)

# SCALING DATA
df_copy = df.copy() # prescaled data
df_scaled = df.drop(['fare_amount']) # no need to scale target value
df_scaled = scale(df_scaled)
cols = df.columns.tolist()
cols.remove('fare_amount') # target col
df_scaled = pd.DataFrame(df_scaled, coulmns=cols, index=df.index)
df_scaled = pd.concat([df_scaled, df['fare_amount']], axis = 1)
df = df_scaled.copy()

# USE SKLEARN TO SPLIT DATA INTO TRAIN & TEST
x = df.loc[:, df.columns != 'fare_amount']
y = df.fare_amount
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# NEURAL NETWORK
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1)

# OUTPUT
train_pred = model.predict(x_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_pred = model.predict(x_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print("Train RMSE: {:0.2f}".format(train_rmse))
print("Test RMSE: {:0.2f}".format(test_rmse))
print("Hello world")
