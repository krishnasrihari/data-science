# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Performing predictions with Linear Regression

# <markdowncell>

# Registering in the required libraries

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model,cross_validation, feature_selection,preprocessing
import statsmodels.formula.api as sm
from statsmodels.tools.eval_measures import mse
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error

# <headingcell level=2>

# Simple Linear Regression

# <markdowncell>

# The following datasets contains the height and weight of a group of Men

# <codecell>

sl_data = pd.read_csv('Data/Mens_height_weight.csv')

# <markdowncell>

# Let's see the distribution of the height and weight

# <codecell>

fig, ax = plt.subplots(1, 1)  
ax.scatter(sl_data['Height'],sl_data['Weight'])
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
plt.show()

# <codecell>

sl_data.corr()

# <markdowncell>

# We'll apply linear regression with keeping Weight as the dependent variable and x as the independent variable

# <codecell>

# Create linear regression object
lm = linear_model.LinearRegression()

# Train the model using the training sets
lm.fit(sl_data.Height[:,np.newaxis], sl_data.Weight)


print 'Intercept is ' + str(lm.intercept_) + '\n'

print 'Coefficient value of the height is ' + str(lm.coef_) + '\n'

pd.DataFrame(zip(sl_data.columns,lm.coef_), columns = ['features', 'estimatedCoefficients'])


# <markdowncell>

# Plotting the regression line on the previous scatter plot

# <codecell>

fig, ax = plt.subplots(1, 1)  
ax.scatter(sl_data.Height,sl_data.Weight)
ax.plot(sl_data.Height,lm.predict(sl_data.Height[:, np.newaxis]), color = 'red')
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
plt.show()

# <headingcell level=2>

# Multiple Regression

# <markdowncell>

# To understand multiple regression, we'll be using the NBA data. The following are the field descriptions
# 
# 1. height = height in feet
# 2. weight = weight in pounds
# 3. success_field_goals = percent of successful field goals (out of 100 attempted)
# 4. success_free_throws = percent of successful free throws (out of 100 attempted)
# 5. avg_points_scored = average points scored per game

# <codecell>

b_data = pd.read_csv('Data/basketball.csv')

b_data.describe()

# <markdowncell>

# Here the average points scored is taken as the dependent variable. We'll see how is the distribution of each of the variable w.r.t the dependent variable

# <codecell>

X_columns = b_data.columns[:-1]

for i in X_columns:
    fig, ax = plt.subplots(1, 1)  
    ax.scatter(b_data[i], b_data.avg_points_scored)
    ax.set_xlabel(i)
    ax.set_ylabel('Average points scored per game')
    plt.show()

# <markdowncell>

# Let's see how each of the variables are correlated with each other

# <codecell>

b_data.corr()

# <markdowncell>

# Let's split the data into train and test where the train set will be used to build the data and the model will be applied to the test set.

# <codecell>

X = b_data.values.copy() 
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split( X[:, :-1], X[:, -1], train_size=0.80)

# <markdowncell>

# We'll generate a Linear Model with the given data 

# <codecell>

result = sm.OLS( y_train, add_constant(X_train) ).fit()
result.summary()

# <markdowncell>

# Since the 3rd variable is significant and the others aren't based on the pvalue. We'll recreate the model only using that variable.

# <codecell>

result_alternate = sm.OLS( y_train, add_constant(X_train[:,2]) ).fit()
result_alternate.summary()

# <markdowncell>

# Lets predict on the test data and see how much is the error 

# <codecell>

ypred = result.predict(add_constant(X_valid))
print mse(ypred,y_valid)

ypred_alternate = result_alternate.predict(add_constant(X_valid[:, 2]))
print mse(ypred_alternate,y_valid)

# <markdowncell>

# Lets see the actual vs predicted for the 1st model

# <codecell>

fig, ax = plt.subplots(1, 1)  
ax.scatter(y_valid, ypred)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

# <markdowncell>

# Same plot for the 2nd plot

# <codecell>

fig, ax = plt.subplots(1, 1)  
ax.scatter(y_valid, ypred_alternate)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

# <codecell>

# Create linear regression object
lm = linear_model.LinearRegression()

# Train the model using the training sets
lm.fit(X_train, y_train)

print 'Intercept is ' + str(lm.intercept_) + '\n'

pd.DataFrame(zip(b_data.columns,lm.coef_), columns = ['features', 'estimatedCoefficients'])

# <markdowncell>

# Lets see how is the R square

# <codecell>

cross_validation.cross_val_score(lm, X_train, y_train, scoring='r2')

# <markdowncell>

# Lets predict the on the test data

# <codecell>

ypred = lm.predict(X_valid)

mean_squared_error(ypred,y_valid)

# <markdowncell>

# Plotting the predicted vs the actual

# <codecell>

fig, ax = plt.subplots(1, 1)  
ax.scatter(y_valid, ypred)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

