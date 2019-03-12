# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Estimating Likelihood with Logisitic Regression

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn import cross_validation, feature_selection,preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from patsy import dmatrices

# <headingcell level=3>

# Titanic Data

# <markdowncell>

# We are going to use the Titanic data that we utilized in chapter 3 for data mining. Following is the description of the columns

# <markdowncell>

# VARIABLE DESCRIPTIONS:
# * <b>survival</b>        Survival (0 = No; 1 = Yes)
# * <b>pclass</b>          Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# * <b>name</b>            Name
# * <b>sex</b>             Sex
# * <b>age</b>             Age
# * <b>sibsp</b>           Number of Siblings/Spouses Aboard
# * <b>parch</b>           Number of Parents/Children Aboard
# * <b>ticket</b>          Ticket Number
# * <b>fare</b>            Passenger Fare
# * <b>cabin</b>           Cabin
# * <b>embarked</b>        Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# <headingcell level=3>

# Inputting the data 

# <codecell>

df = pd.read_csv('Data/titanic data.csv')

df

# <markdowncell>

# Counting the missing values in each column

# <codecell>

df.count(0)

# <markdowncell>

# Understanding the field descriptions of the data and seeing the missing value of the data. We can see that Ticket and Cabin column won't add much value to the model building process as the ticket column is basically unique identifier for each passenger and the Cabin column is mostly empty.
# 
# We'll remove these two columns from our dataframe

# <codecell>

df = df.drop(['Ticket','Cabin','Name'], axis=1)

# Remove missing values
df = df.dropna() 

df

# <headingcell level=2>

# Model Building

# <markdowncell>

# Let's split the into training and testing set

# <codecell>


df_train = df.iloc[ 0: 600, : ]
df_test = df.iloc[ 600: , : ]


# <markdowncell>

# Let's build a logisitc regression model using statsmodel

# <codecell>

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked) +  Parch' 

y_train,x_train = dmatrices(formula, data=df_train, return_type='dataframe')
y_test,x_test = dmatrices(formula, data=df_test, return_type='dataframe')

# instantiate our model
model = sm.Logit(y_train,x_train)
res = model.fit()
res.summary()

# <markdowncell>

# Based on the significant variables, let's remodel it

# <codecell>

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp ' 

y_train,x_train = dmatrices(formula, data=df_train, return_type='dataframe')
y_test,x_test = dmatrices(formula, data=df_test, return_type='dataframe')

# instantiate our model
model = sm.Logit(y_train,x_train)
res = model.fit()
res.summary()

# <markdowncell>

# Lets see prediction distribution 

# <codecell>

kde_res = KDEUnivariate(res.predict())
kde_res.fit()
plt.plot(kde_res.support,kde_res.density)
plt.fill_between(kde_res.support,kde_res.density, alpha=0.2)
plt.title("Distribution of our Predictions")

# <markdowncell>

# Distribution of prediction based on the Gender

# <codecell>

plt.scatter(res.predict(),x_train['C(Sex)[T.male]'] , alpha=0.2)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted chance of survival")
plt.ylabel("Male Gender")
plt.title("The Change of Survival Probability by Gender being Male")

# <markdowncell>

# Distribution of the prediction based on the lower class

# <codecell>

plt.scatter(res.predict(),x_train['C(Pclass)[T.3]'] , alpha=0.2)
plt.xlabel("Predicted chance of survival")
plt.ylabel("Class Bool")
plt.grid(b=True, which='major', axis='x')
plt.title("The Change of Survival Probability by Lower Class which is 3rd class")

# <markdowncell>

# Distribution of prediction based on Age

# <codecell>

plt.scatter(res.predict(),x_train.Age , alpha=0.2)
plt.grid(True, linewidth=0.15)
plt.title("The Change of Survival Probability by Age")
plt.xlabel("Predicted chance of survival")
plt.ylabel("Age")

# <markdowncell>

# Distribution of prediction based on number of siblings

# <codecell>

plt.scatter(res.predict(),x_train.SibSp , alpha=0.2)
plt.grid(True, linewidth=0.15)
plt.title("The Change of Survival Probability by Number of siblings/spouses")
plt.xlabel("Predicted chance of survival")
plt.ylabel("No. of Siblings/Spouses")

# <markdowncell>

# Let's predict using the model on the test data  and see how is the performance of the model throught the precision and recall by keeping a threshold of 0.7

# <codecell>

y_pred = res.predict(x_test)
y_pred_flag = y_pred > 0.7
print pd.crosstab(y_test.Survived
                  ,y_pred_flag
                  ,rownames = ['Actual']
                  ,colnames = ['Predicted'])

print '\n \n'

print classification_report(y_test,y_pred_flag)

# <markdowncell>

# Let's compute the receiver operating characteristics

# <codecell>

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

# <markdowncell>

# Let's plot the roc curve

# <codecell>

# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# <markdowncell>

# Let's build the same model using scikit 

# <codecell>

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(x_train, y_train.Survived)

# <codecell>

# examine the coefficients
pd.DataFrame(zip(x_train.columns, np.transpose(model.coef_)))

# <markdowncell>

# Let's check out the precision and recall for it

# <codecell>

y_pred = model.predict_proba(x_test)
y_pred_flag = y_pred[:,1] > 0.7


print pd.crosstab(y_test.Survived
                  ,y_pred_flag
                  ,rownames = ['Actual']
                  ,colnames = ['Predicted'])

print '\n \n'

print classification_report(y_test,y_pred_flag)

# <markdowncell>

# Let's compute the roc curve

# <codecell>

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1])
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

# <markdowncell>

# Let's plot the roc curve

# <codecell>

# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

