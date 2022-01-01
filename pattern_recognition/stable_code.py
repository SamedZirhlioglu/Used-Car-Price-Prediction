import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import read_csv
from scipy.stats import zscore
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def diff_values(valid, pred):
    diff_data = (pred - valid).values.tolist()
    return diff_data

def range_gen(list):
    return range(1, len(list) + 1)

def plot_result(model_name, valid, pred, r2, mse, rmse, color):
    plt.subplots(figsize=(35,10), dpi=100)
    plt.bar(range_gen(valid), height=valid, label='Valid Data', color=color, zorder=1)

    plt.plot(range_gen(pred), pred, label='Pred Data', color='b', zorder=2)
    plt.scatter(range_gen(pred), pred, label='Pred Data', color='black', zorder=3)
    plt.legend(loc='lower right')
    plt.title(
        model_name +
        '\n$R^2$ Score = ' + str(r2) +
        '\nMSE = ' + str(mse) +
        '\nRMSE = ' + str(rmse)
    )
    plt.show()

def plot_errors(model_name, pred_1, valid_1, pred_2, valid_2):
    plt.scatter(
        pred_1,
        diff_values(valid_1, pred_1),
        color='orange', s=10, label='Train data'
    )
    plt.scatter(
        pred_2,
        diff_values(valid_2, pred_2),
        color='blue', s=10, label='Test data'
    )
    plt.hlines(y = 0, xmin = 0, xmax = valid_1.max(), linewidth = 2)
    plt.legend(loc = 'lower right')
    plt.title(model_name + ' Residual Errors')
    plt.show()

# Veri seti import edildi ve DataFrame'e aktarıldı.
data = read_csv('./stable_data.csv')
df = pd.DataFrame(data)
min_max_scaler = preprocessing.MinMaxScaler()
scaled_df = min_max_scaler.fit_transform(df)
df = pd.DataFrame(scaled_df)

# Veri seti train-test olarak %80-%20 oranında bölündü.
X = df.drop(4, axis=1)
y = df[4]

TEST_SIZE = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE)
valid_data = y_test.values.tolist()

### ################ ###
### LinearRegression ###
### ################ ###

# Linear Regression modeli oluşturuldu ve eğitildi.
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

# Eğitilen modelin sonuçları alındı.
coef_linear = model_linear.coef_
r2_linear = model_linear.score(X_test, y_test)
pred_linear = model_linear.predict(X_test).ravel().tolist()
mse_linear = mean_squared_error(valid_data, pred_linear)
rmse_linear = (np.sqrt(mean_squared_error(valid_data, pred_linear)))

# Sonuçlar konsola yazdırıldı
print("Coef: " + str(coef_linear))
print("R2: " + str(r2_linear))
print("MSE: " + str(mse_linear))
print("RMSE: " + str(rmse_linear))

# Veriler görselleştirildi
plot_result('Linear Regression', valid_data, pred_linear, r2_linear, mse_linear, rmse_linear, 'pink')
plot_errors('Linear Regression', model_linear.predict(X_train), y_train, model_linear.predict(X_test), y_test)