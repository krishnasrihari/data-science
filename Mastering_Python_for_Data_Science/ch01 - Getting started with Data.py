
# coding: utf-8

# # Getting started with Data

# In[1]:

import numpy as np
import pandas as pd
import json


# ## The world of Arrays with Numpy

# Arrays are fundamental in processing data. Arrays are utilized to represent matrices which is heavily used in mathematics. Python by default comes with data structure like List which can be utilized for array operation but a python list on its own is not suitable to perform heavy mathematical operation as it is not optimized for it. 
# Numpy is a wonderful python package by Travis Oliphant which has been created fundamentally for scientific computing. It helps in handling large  multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays

# ### Creating an Array

# In[2]:

n_array = np.array([[0, 1, 2, 3],
                 [4, 5, 6, 7],
                 [8, 9, 10, 11]])


# The shape gives the size of each dimension of the array

# In[3]:

n_array.shape


# The number of axes of the array

# In[4]:

n_array.ndim


# The number of elements

# In[5]:

n_array.size


# The datatype of the elements in the array

# In[6]:

n_array.dtype.name


# ### Mathematical Operations

# When you have an array of data, you would like to perform certain mathematical operations on it. Below are few of the important ones

# Subtracting two arrays

# In[7]:

a = np.array([11, 12, 13, 14])
b = np.array([1, 2, 3, 4])
c = a - b
c


# Squaring the array

# In[8]:

b ** 2


# Trignometric function on the array

# In[10]:

np.cos(b)


# Conditional Operation

# In[11]:

b < 2


# Matrix Multiplication - Element wise product and dot product

# In[13]:

A1 = np.array([[1, 1],
            [0, 1]])

A2 = np.array([[2, 0],
            [3, 4]])

A1 * A2
np.dot(A1, A2)


# ### Indexes and slicing  Selecting a single element

# In[13]:

n_array[0, 1]


# select range of values in a row

# In[14]:

n_array[0, 0:3]


# selecting an entire row of values

# In[15]:

n_array[0, :]


# selecting an entire column of values

# In[17]:

n_array[:, 1]


# Changing the shape of the array

# In[14]:

n_array.ravel()

n_array.shape = (6, 2)
n_array



# In[15]:

n_array.transpose()


# ## Empowering Data Analysis with Pandas

# Pandas is an open source Python library specially designed for Data Analysis. It has been built on numpy and makes it easy to handle data. Numpy is a fairly low level tool which handles matrices really well and it can be seen similar to Matlab.
# Pandas brings the richness of R in the world of python to handle the data. It’s got efficient data structures to process the data, performs fast joins, read data from various sources to name a few.

# Pandas Data Structure

# Pandas has essentially three data structures 
# 1. Series
# 2. Data Frame
# 3. Panel

# ### Series

# Creating series from random numbers

# In[17]:

pd.Series(np.random.randn(5))


# Creating series with row labels

# In[18]:

pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])


# Creating series from dict

# In[19]:

d = {'A': 10, 'B': 20, 'C': 30}  
pd.Series(d)


# ### DataFrame

# Creating data frame from a dict of Series

# In[20]:

d = {'c1': pd.Series(['A', 'B', 'C']),
     'c2': pd.Series([1, 2., 3., 4.])}

df = pd.DataFrame(d)
df


# Creating data frame from a dict of Lists

# In[21]:

d = {'c1': ['A', 'B', 'C', 'D'],
     'c2': [1, 2., 3., 4.]}

df = pd.DataFrame(d)
df


# ### Panel

# Creating a Panel Data

# In[22]:

d = {'Item1': pd.DataFrame(np.random.randn(4, 3)),
     'Item2': pd.DataFrame(np.random.randn(4, 2))}

pd.Panel(d)


# ## Inputting and Exporting Data

# The data is persisted in various forms like csv,tsv, databases etc. Pandas makes it convenient to read data from these formats or to export to these formats. We’ll use a dataset which contains the weight statistics of the school students from United states of America

# ### CSV

# To read data from a csv file, the following command can be used

# In[9]:

d = pd.read_csv('Data/Student_Weight_Status_Category_Reporting_Results__Beginning_2010.csv')
d[0:5]['AREA NAME']


#  Writing data to a csv

# In[23]:

d = {'c1': pd.Series(['A', 'B', 'C']),
     'c2': pd.Series([1, 2., 3., 4.])}

df = pd.DataFrame(d)
df.to_csv('sample_data.csv')


# ### Excel

#  Reading data from an excel file

# In[24]:

d = pd.read_excel('Data/Student_Weight_Status_Category_Reporting_Results__Beginning_2010.xls')
d[0:5]


# Writing to Excel File

# In[25]:

df.to_excel('sample_data.xls')


# ### JSON

# Reading from json

# In[26]:

json_data = open('Data/Student_Weight_Status_Category_Reporting_Results__Beginning_2010.json')
data = json.load(json_data)
json_data.close()


# ## Data Cleansing

# ### Handling missing data

# checking if an instance is null

# In[27]:

d = pd.read_csv('Data/Student_Weight_Status_Category_Reporting_Results__Beginning_2010.csv')
d['Location 1'].isnull()


# Aggregating to count the null instances

# In[28]:

d['Location 1'].isnull().value_counts()


# removing the missing values

# In[39]:

d['Location 1'].dropna()


# Removing all na instances

# In[40]:

d = d.dropna(how='any')


# ### Filling the missing data

# Let's create a dataframe with some values

# In[32]:

df = pd.DataFrame(np.random.randn(5, 3), index=['a0', 'a10', 'a20', 'a30', 'a40'],
                  columns=['X', 'Y', 'Z'])
df


# Let's add some more values which will have null values

# In[34]:

df2 = df.reindex(['a0', 'a1', 'a10', 'a11', 'a20', 'a21', 'a30', 'a31', 'a40','a41'])
df2


# Filling with a zero value

# In[36]:

df2.fillna(0)


# Filling with forward propagation

# In[38]:

df2.fillna(method='pad')


# Filling with the mean of the column

# In[40]:

df2.fillna(df2.mean())


# ## String Manipulation

# There are character field columns which you would want to modify. The following techniques explains how to do it

# ### Substring

# Choosing five rows of the Area column in the data

# In[42]:

df = pd.read_csv('Data/Student_Weight_Status_Category_Reporting_Results__Beginning_2010.csv')
df['AREA NAME'][0:5]


# In order to extract the 1st word from the Area Name column

# In[19]:

df['AREA NAME'][0:5].str.extract('(\w+)')


# Getting the 1st two words

# In[44]:

df['AREA NAME'][0:5].str.extract('(\w+)\s(\w+)')


# ### Uppercase

# In[21]:

df['AREA NAME'][0:5].str.upper()


# ### Lowercase

# In[22]:

df['AREA NAME'][0:5].str.lower()


# ### Length

# In[23]:

df['AREA NAME'][0:5].str.len()


# ### Split

# In[24]:

df['AREA NAME'][0:5].str.split(' ')


# ### Replace

# To replace all the Area Names ending with DISTRICT to DIST

# In[25]:

df['AREA NAME'][0:5].str.replace('DISTRICT$', 'DIST')


# ### Merging Data

# To combine data sets together, the concat function of Pandas can be utilized.
# Let’s take the Area Name and County with its first five rows

# In[26]:

df[['AREA NAME', 'COUNTY']][0:5]


# In[29]:

p1 = df[['AREA NAME', 'COUNTY']][0:2]
p2 = df[['AREA NAME', 'COUNTY']][2:5]


# In[31]:

pd.concat([p1,p2])


# The combined pieces can be identified by assigning a key

# In[32]:

concatenated = pd.concat([p1,p2], keys = ['p1','p2'])
concatenated


# Using the keys, the pieces can be extracted back from concatenated data 

# In[33]:

concatenated.ix['p1']


# ## Aggregation Operations

# There are number of aggregation operations like average, sum etc that you would like to do on a numerical field. Following explains on how to perform it 

# ### Average

# To find out the average number of students in Elementary school who are Obese

# In[38]:

data = df[df['GRADE LEVEL'] == 'ELEMENTARY']
data['NO. OBESE'].mean()


# ### SUM

# To find out the total number of Elementary students who are obese across all the school

# In[39]:

data['NO. OBESE'].sum()


# ### MAX

# To get the maximum number of students that are Obese in an elementary school

# In[36]:

data['NO. OBESE'].max()


# ### MIN

# To get the minimum number of students that are Obese in an elementary school

# In[40]:

data['NO. OBESE'].min()


# ### STD

# To get the standard deviation of the number of Obese students

# In[41]:

data['NO. OBESE'].std()


# ### COUNT

# To count the total number of schools with Elementary grade in the Columbia county

# In[42]:

data = df[(df['GRADE LEVEL'] == 'ELEMENTARY') & (df['COUNTY'] == 'DELAWARE')]
data['COUNTY'].count()


# ## Joins

# SQL like joins can be performed on the data frame using pandas.
# Lets define a lookup dataframe which assigns levels to each of the grade

# In[45]:

grade_lookup = {'GRADE LEVEL': pd.Series(['ELEMENTARY', 'MIDDLE/HIGH', 'MISC']),'LEVEL': pd.Series([1, 2, 3])}
grade_lookup = pd.DataFrame(grade_lookup)


# Lets take the 1st five rows of the  grade data column as an example for performing the joins

# In[44]:

df[['GRADE LEVEL']][0:5]


# ### Inner Join

# An inner join can be performed with the following command

# In[46]:

d_sub = df[0:5].join(grade_lookup.set_index(['GRADE LEVEL']), on=['GRADE LEVEL'], how='inner')
d_sub[['GRADE LEVEL', 'LEVEL']]


# ### Left Outer Join

# A left outer join can be performed with the following commands

# In[46]:

d_sub = df[0:5].join(grade_lookup.set_index(['GRADE LEVEL']), on=['GRADE LEVEL'], how='left')
d_sub[['GRADE LEVEL', 'LEVEL']]


# ### Full outer join

# The full outer join can be performed with the following code

# In[47]:

d_sub = df[0:5].join(grade_lookup.set_index(['GRADE LEVEL']), on=['GRADE LEVEL'], how='outer')
d_sub[['GRADE LEVEL', 'LEVEL']]


# ### Group By

# Its easy to do an SQL like group by operation with Pandas. Lets say, if you want to find the sum of the number of Obese students in each of the grade then  you can use the following command

# In[48]:

df['NO. OBESE'].groupby(d['GRADE LEVEL']).sum()


# This command chooses the Number of Obese students column and then uses the  group by method to group the data based group level and finally the sum method sums up the number. The same can be achieved by the following function too

# In[49]:

df['NO. OBESE'].groupby(d['GRADE LEVEL']).aggregate(sum)


# Here the aggregate method is utilized. The sum function is passed to obtain the required results.
# It’s also possible to obtain multiple kinds of aggregation on the same metric. This can be achieved by the following command

# In[50]:

df['NO. OBESE'].groupby(d['GRADE LEVEL']).aggregate([sum, mean, std])

