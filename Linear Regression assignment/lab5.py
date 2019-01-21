import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

hw1 = pd.read_csv("./height_weight1.csv")
hw2 = pd.read_csv("./height_weight2.csv")
car = pd.read_csv("./car.csv")

hw1_train, hw1_test = train_test_split(hw1, test_size=.3, random_state=123)
hw2_train, hw2_test = train_test_split(hw2, test_size=.3, random_state=123)
car_train, car_test = train_test_split(car, test_size=.3, random_state=123)

# ----------
# Question 1
# ----------

# create two models, one with intercept and one without
hw1_model = smf.ols(formula='weight ~ height', data=hw1_train).fit()
hw1_model_no_intercept = smf.ols(
    formula='weight ~ height - 1', data=hw1_train).fit()
hw1_model.summary()
hw1_model_no_intercept.summary()

# how do these compare in terms of residual assumptions?
hw1_residuals = hw1_model.resid
hw1_residuals_no_intercept = hw1_model_no_intercept.resid

# 1. error variable is normally distributed
plt.hist(hw1_residuals)
plt.show()
plt.hist(hw1_residuals_no_intercept)
plt.show()
# both of these models are relatively centered with a mean of 0.

# 2. error variance is constant for all xs
plt.plot(hw1_model.predict(hw1_train), hw1_residuals, '.')
plt.show()
plt.plot(hw1_model_no_intercept.predict(
    hw1_train), hw1_residuals_no_intercept, '.')
plt.show()
# with an intercept, the assumption of a constant variance is present
# this plot is very scattered, which is good for residuals
# without an intercept, heteroscedasticity is present
# the data is plotted in a very narrow cone, which is not good
# it means there is something missing

# 3. errors are independent of each other
# I'm not sure how to plot these over time or if these models
# differ in this respect.

# There are 2 things that tell us that it's much better to use the
# model with an intercept - assumption 2, which shows that the
# intercept model has no pattern to its residuals, a good thing;
# and the R^2 values, which are 0.992 for non-intercept and 0.373
# for the intercept - a clear difference.

# ----------
# Question 2
# ----------

hw2_model = smf.ols(formula='weight ~ height - 1', data=hw2_train).fit()
hw2_model.summary()

# Does this model meet the assumptions?
hw2_residuals = hw2_model.resid

# 1. normal distribution?
plt.hist(hw2_residuals)
plt.show()
# yes, it is centered around 0.

# 2. constant variance?
plt.plot(hw2_model.predict(hw2_train), hw2_residuals, '.')
plt.show()
# no, this exhibits heteroscedasticity or patterns in its variance

# 3. independence over time?
# Again not totally sure how to plot this, but it already
# doesn't meet assumption 2 so it doesn't matter.

# How does this effect impact our prediction?
# it will make our predictions incorrect, especially for certain
# parts of our model. in this case, it could affect parts where
# the height was especially high or low. It would throw off some
# point predictions, and mess with the prediction intervals.

# ----------
# Question 3
# ----------

car_model1 = smf.ols(formula='Price ~ 1 + Miles + Age', data=car_train).fit()
car_model1.summary()

car_model2 = smf.ols(
    formula='Price ~ Miles + Age + Make + Type - 1', data=car_train).fit()
car_model2.summary()

car_model3 = smf.ols(
    formula='Price ~ 1 + Miles + Age + C(Make) + C(Type)', data=car_train).fit()
car_model3.summary()

# I consider model 3 to be the best, since it takes into account all the
# relevant variables and properly uses the categorical ones.
# It has a slightly higher R^2 value, but the Durbin-Watson is 1.998 and the
# coefficients are all relatively similar instead of very disparate.

# 7 yr old, BMW, 67000 miles, convertible
point = pd.DataFrame.from_dict(
    {'Age': [7], 'Make': ['BMW'], 'Type': [3], 'Miles': [67000]})
predicted = car_model3.predict(point)
predicted
# the predicted price is $25,664.99
