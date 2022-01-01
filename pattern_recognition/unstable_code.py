import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

csv = read_csv('./unstable_data.csv')
dataframe = pd.DataFrame(csv)
#dataframe = zscore(dataframe)

scaler = MinMaxScaler()
dataframe = scaler.fit_transform(dataframe)
# Veri seti train-test olarak %70-%30 oranında bölündü.
X = dataframe[:,:-1]
y = dataframe[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("r2:" + str(metrics.r2_score(y_test, prediction)))
print("mean_absolute_error:" + str(metrics.mean_absolute_error(y_test, prediction)))
print("mean_squared_error:" + str(metrics.mean_squared_error(y_test, prediction)))
print("root_mean_squared_error:" + str(math.sqrt(metrics.mean_squared_error(y_test, prediction))))

plt.plot(range(len(y_test)), y_test, label='Valid')
plt.plot(range(len(prediction)), prediction, label='Tahmin')
plt.legend(loc='lower right')
plt.title("Tahmin & Valid Değerler")
plt.show()

"""plt.figure(figsize=(13,6))
ax1 = sns.distplot(y_test, color = 'purple', label='Valid')
ax2 = sns.distplot(prediction, color = 'orange', label='Pred')
plt.legend(loc='lower right')
plt.title("Distribution of age")
plt.show()"""