from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./data.csv")

dataframe = pd.DataFrame(data)

scaler = MinMaxScaler()
dataframe = pd.DataFrame(scaler.fit_transform(dataframe))

X = dataframe.drop([4], axis=1)
y = dataframe[4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#Değerlerden kaç tane olduğu
y_train.value_counts()

xgbModel = XGBRegressor(random_state=42)
xgbModel.fit(X_train,y_train)
#Tahminleri alma
y_predicted = xgbModel.predict(X_test)

np.array(y_test)

xgb_r2 = r2_score(y_test, y_predicted)
xgb_MSE= mean_squared_error(y_test,y_predicted, squared=(True))
xgb_RMSE = mean_squared_error(y_test,y_predicted, squared=(False))

print('R2:', xgb_r2)
print('MSE:', xgb_MSE)
print('RMSE:', xgb_RMSE)

plt.plot(range(len(y_predicted)), y_predicted)
plt.plot(range(len(y_test)), y_test)

plt.title("Pred-Valid")
plt.show()