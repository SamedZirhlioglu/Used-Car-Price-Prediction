from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
TEST_SIZE = 0.3


df = pd.read_csv("./data.csv")
print(df.info())
df=pd.DataFrame(df)
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

X=df[:,:-1]
y=df[:,-1:]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=42)

# RANDOM FOREST TREE REGRESSOR
randomForestTreeModel = RandomForestRegressor(random_state=42)
randomForestTreeModel.fit(X_train, y_train)
Ypred_randomForestTree = randomForestTreeModel.predict(X_test)

score = randomForestTreeModel.score(X_test, y_test)
print("ACC score: " + str(score))

mse = mean_squared_error(y_test, Ypred_randomForestTree)
print("MSE: " + str(mse))
print("RMSE: " + str(np.sqrt(mse)))                                                               
print("R2: " + str(r2_score(y_test,Ypred_randomForestTree)))

plt.figure(figsize=(13,6))
ax1 = sns.distplot(y_test, color = 'purple', label='Valid')
ax2 = sns.distplot(Ypred_randomForestTree, color = 'orange', label='Pred')
plt.legend(loc='upper right')
plt.title("Valid-Prediciton Fark Karşılaştırması")
plt.show()