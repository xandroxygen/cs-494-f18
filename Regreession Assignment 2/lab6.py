import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing

data = pd.read_csv("./icudata.csv")
data.drop(['ID', 'SEX'], axis=1, inplace=True)
# STA is our response var


# Logistic Regression
X = data.ix[:, (1, 2, 3, 4, 5, 6)].values
y = data.ix[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=25)

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)
y_prob = LogReg.predict_proba(X_test)

# 1a. Table of coefficients
coef_table = pd.DataFrame(
    list(zip(list(data)[1:], LogReg.coef_[0]))).transpose()
print(coef_table)

# 1b. Odds of survival with CPR
# You are 33% more likely to survive if you receive CPR.
odds = 1 / (1 + np.exp(LogReg.coef_))
odds_table = pd.DataFrame(list(zip(list(data)[1:], odds[0]))).transpose()
print(odds_table)

# LASSO Regression

predictors = data[['AGE', 'RACE', 'CPR', 'SYS', 'HRA', 'TYP']].copy()
target = data.STA

# standardize predictors
predictors['AGE'] = preprocessing.scale(predictors['AGE'].astype('float64'))
predictors['RACE'] = preprocessing.scale(predictors['RACE'].astype('float64'))
predictors['CPR'] = preprocessing.scale(predictors['CPR'].astype('float64'))
predictors['SYS'] = preprocessing.scale(predictors['SYS'].astype('float64'))
predictors['HRA'] = preprocessing.scale(predictors['HRA'].astype('float64'))
predictors['TYP'] = preprocessing.scale(predictors['TYP'].astype('float64'))

# split data into train and test sets
pred_train, pred_test, resp_train, resp_test = train_test_split(
    predictors, target, test_size=.3, random_state=123)

model = LassoLarsCV(cv=10, precompute=True).fit(pred_train, resp_train)

# 1c. optimal alpha value = 0.0007406043842015065
print(model.alpha_)

# 1d. Table of coefficients

coef_table = pd.DataFrame(
    list(zip(list(data)[1:], model.coef_))).transpose()
print(coef_table)
