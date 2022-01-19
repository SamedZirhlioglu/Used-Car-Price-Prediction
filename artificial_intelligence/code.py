from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import zscore
from matplotlib import cm
from matplotlib.colors import Normalize

sns.set(font_scale=1.5)

# Veri seti tanımlandı
data = pd.read_csv('data.csv')
df = zscore(pd.DataFrame(data))

# Veri seti X-y olarak ayrıldı.
X = df.drop(['price'], axis=1)
y = df['price']

# Veri seti train-test olarak (%70-30) ayrıldı.
data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Eğitim verisinin değerlerini görme
print(data.price.describe())

# Fiyata göre histogram grafiği oluşturulması
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": (0.75, 0.15)})
sns.histplot(data=data_train, x='price', ax=ax[0], kde=True)
sns.boxplot(data=data_train, x='price', ax=ax[1])
plt.tight_layout()
plt.show()

# Algoritma sonuçlarının tutulacağı List'lerin tanımlanması
names = []
preds = []
scores = []
cross_scores = []

print("\n\n\n")
#################################################################################################################################################
### DECISION TREE REGRESSOR #####################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
decisionTreeModel = DecisionTreeRegressor(random_state=42)
decisionTreeModel.fit(X_train, y_train)
Ypred_decisionTree = decisionTreeModel.predict(X_test)

names.append("Decision Tree Regressor")
preds.append(Ypred_decisionTree)
scores.append(r2_score(Ypred_decisionTree, y_test))
cross_scores.append(cross_val_score(decisionTreeModel, X_train, y_train, scoring='r2', cv=30))

plt.plot(cross_scores[0])
plt.title(names[0] + " Cross-Validation Scores")
plt.show()

plt.plot(range(len(preds[0][:250])), preds[0][:250], label=names[0] + ' Pred')
plt.plot(range(len(y_test[:250])), y_test[:250], label='Valid')
plt.legend(loc='lower right')
plt.title(label=names[0] + " Tahmin & Valid Değerleri")
plt.show()

#################################################################################################################################################
### GRADIENT BOOSTING REGRESSOR #################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
gradientBoostingModel = GradientBoostingRegressor(random_state=42)
gradientBoostingModel.fit(X_train, y_train)
Ypred_gradientBoosting = gradientBoostingModel.predict(X_test)

names.append("Gradient Boosting Regressor")
preds.append(Ypred_gradientBoosting)
scores.append(r2_score(Ypred_gradientBoosting, y_test))
cross_scores.append(cross_val_score(gradientBoostingModel, X_train, y_train, scoring='r2', cv=30))

plt.plot(cross_scores[1])
plt.title(names[1] + " Cross-Validation Scores")
plt.show()

plt.plot(range(len(preds[1][:250])), preds[1][:250], label=names[1] + ' Pred')
plt.plot(range(len(y_test[:250])), y_test[:250], label='Valid')
plt.legend(loc='lower right')
plt.title(label=names[1] + " Tahmin & Valid Değerleri")
plt.show()

#################################################################################################################################################
### XGB REGRESSOR ###############################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
xgbModel = XGBRegressor(random_state=42)
xgbModel.fit(X_train, y_train)
Ypred_xgb = xgbModel.predict(X_test)

names.append("XGB Regressor")
preds.append(Ypred_xgb)
scores.append(r2_score(Ypred_xgb, y_test))
cross_scores.append(cross_val_score(xgbModel, X_train, y_train, scoring='r2', cv=30))

plt.plot(cross_scores[2])
plt.title(names[2] + " Cross-Validation Scores")
plt.show()

plt.plot(range(len(preds[2][:250])), preds[2][:250], label=names[2] + ' Pred')
plt.plot(range(len(y_test[:250])), y_test[:250], label='Valid')
plt.legend(loc='lower right')
plt.title(label=names[2] + " Tahmin & Valid Değerleri")
plt.show()

#################################################################################################################################################
### RANDOM FOREST REGRESSOR #####################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
randomForestModel = RandomForestRegressor(random_state=42)
randomForestModel.fit(X_train, y_train)
Ypred_randomForest = randomForestModel.predict(X_test)

names.append("Random Forest Regressor")
preds.append(Ypred_randomForest)
scores.append(r2_score(Ypred_randomForest, y_test))
cross_scores.append(cross_val_score(randomForestModel, X_train, y_train, scoring='r2', cv=30))

plt.plot(cross_scores[3])
plt.title(names[3] + " Cross-Validation Scores")
plt.show()

plt.plot(range(len(preds[3][:250])), preds[3][:250], label=names[3] + ' Pred')
plt.plot(range(len(y_test[:250])), y_test[:250], label='Valid')
plt.legend(loc='lower right')
plt.title(label=names[3] + " Tahmin & Valid Değerleri")
plt.show()

#################################################################################################################################################
### LGBM REGRESSOR ##############################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
lgbmModel = LGBMRegressor(random_state=42)
lgbmModel.fit(X_train, y_train)
Ypred_lgbm = lgbmModel.predict(X_test)

names.append("LGBM Regressor")
preds.append(Ypred_lgbm)
scores.append(r2_score(Ypred_lgbm, y_test))
cross_scores.append(cross_val_score(lgbmModel, X_train, y_train, scoring='r2', cv=30))

plt.plot(cross_scores[4])
plt.title(names[4] + " Cross-Validation Scores")
plt.show()

plt.plot(range(len(preds[4][:250])), preds[4][:250], label=names[4] + ' Pred')
plt.plot(range(len(y_test[:250])), y_test[:250], label='Valid')
plt.legend(loc='lower right')
plt.title(label=names[4] + " Tahmin & Valid Değerleri")
plt.show()

#################################################################################################################################################
### CAT BOOST REGRESSOR #########################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
catBoostModel = CatBoostRegressor(verbose=0, random_state=42)
catBoostModel.fit(X_train, y_train)
Ypred_catBoost = catBoostModel.predict(X_test)

names.append("Cat Boost Regressor")
preds.append(Ypred_catBoost)
scores.append(r2_score(Ypred_catBoost, y_test))
cross_scores.append(cross_val_score(catBoostModel, X_train, y_train, scoring='r2', cv=30))

plt.plot(cross_scores[5])
plt.title(names[5] + " Cross-Validation Scores")
plt.show()

plt.plot(range(len(preds[5][:250])), preds[5][:250], label=names[5] + ' Pred')
plt.plot(range(len(y_test[:250])), y_test[:250], label='Valid')
plt.legend(loc='lower right')
plt.title(label=names[5] + " Tahmin & Valid Değerleri")
plt.show()

for i in range(len(cross_scores)):
    cross_scores[i].sort()

#################################################################################################################################################
### SONUÇLARIN GÖRSELLEŞTİRİLMESİ ###############################################################################################################
#################################################################################################################################################
#################################################################################################################################################

def plot_scores():
    fig, axes = plt.subplots(figsize=(5,5))
    plt.bar(names, height=scores)
    plt.title('BMW Pricing')
    plt.show()

    print("\n\nALL SCORES")
    for i in range(0, len(names)):
        print(str(names[i]) + ": " + str(scores[i]))

def plot_cross_scores():
    std_cross_scores = []
    mean_cross_scores = []
    for score in cross_scores:
        std_cross_scores.append(np.std(score))
        mean_cross_scores.append(np.mean(score))
    
    fig, axes = plt.subplots(figsize=(5,5))
    plt.bar(names, height=mean_cross_scores)
    plt.title('Cross-Scores (Mean)')
    plt.show()

    print("\n\nALL CROSS VALIDATON SCORES (MEAN)")
    for i in range(0, len(names)):
        print(str(names[i]) + " Cross-Validation Score: " + str(mean_cross_scores[i]))
        print(str(names[i]) + " Standard Deviation Score: " + str(std_cross_scores[i]))
    
def log_MSEs():
    mse = []
    print("\n\nALL MSE SCORES")
    for i in range(len(preds)):
        mse.append(mean_squared_error(y_test, preds[i]))
        print(str(names[i]) + ": " + str(mse[i]))
    fig, axes = plt.subplots(figsize=(35,10), dpi=100)
    plt.bar(names, height=mse)
    plt.title("MSE Values")
    plt.show()

def plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    AX = [1,2,3,4,5,6]
    x = np.array([[i] * 30 for i in range(1, 7)]).ravel() #algoritma çalışma sayısı
    y = np.array([i for i in range(1, 31)] * len(AX))
    z = np.zeros(len(AX)*30)

    dx = np.ones(len(AX)*30) # length along x-axis of each bar
    dy = np.ones(len(AX)*30) # length along y-axis of each bar
    dz = np.array(cross_scores).ravel() # length along z-axis of each bar (height)

    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=min(dz), vmax=max(dz))
    colors = cmap(norm(dz))
    sc = cm.ScalarMappable(cmap=cmap,norm=norm)
    sc.set_array([])
    plt.colorbar(sc)

    ax.bar3d(x, y, z, dx, dy, dz, color=colors, zsort='average')
    ax.set_ylabel('Cross-Validation Çalışmaları')
    ax.set_zlabel('Cross-Validation Çıktıları')

    plt.show()

plot_scores()
plot_cross_scores()
log_MSEs()
plot_3d()