from pandas import *
from pylab import *

# Reading the titanic data
d = read_csv('Data/titanic data.csv')

# Which passenger class had the maximum survivors?
d['Pclass'].isnull().value_counts()
d['Survived'].isnull().value_counts()

survivors = d.groupby('Pclass')['Survived'].agg(sum)
total_passengers = d.groupby('Pclass')['PassengerId'].count()
survivor_percentage = survivors / total_passengers

fig = plt.figure()
ax = fig.add_subplot(111)

rect = ax.bar(survivors.index.values.tolist(), survivors, color='blue', width=0.5)
ax.set_ylabel('No. of survivors')
ax.set_title('Total number of survivors based on class')
xTickMarks = survivors.index.values.tolist()
ax.set_xticks(survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

rect = ax.bar(survivor_percentage.index.values.tolist(), survivor_percentage, color='blue', width=0.5)
ax.set_ylabel('Survivor Percentage')
ax.set_title('Percentage of survivors based on class')
xTickMarks = survivors.index.values.tolist()
ax.set_xticks(survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()



# What was distribution based on gender for the survivors among the class
d['Sex'].isnull().value_counts()

male_survivors = d[d['Sex'] == 'male'].groupby('Pclass')['Survived'].agg(sum)
male_total_passengers = d[d['Sex'] == 'male'].groupby('Pclass')['PassengerId'].count()
male_survivor_percentage = male_survivors / male_total_passengers

female_survivors = d[d['Sex'] == 'female'].groupby('Pclass')['Survived'].agg(sum)
female_total_passengers = d[d['Sex'] == 'female'].groupby('Pclass')['PassengerId'].count()
female_survivor_percentage = female_survivors / female_total_passengers

fig = plt.figure()
ax = fig.add_subplot(111)
index = np.arange(male_survivors.count())
bar_width = 0.35

rect1 = ax.bar(index, male_survivors, bar_width, color='blue', label='Men')
rect2 = ax.bar(index + bar_width, female_survivors, bar_width, color='y', label='Women')

ax.set_ylabel('Survivor Numbers')
ax.set_title('Male and Female survivors based on class')
xTickMarks = male_survivors.index.values.tolist()
ax.set_xticks(index + bar_width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
index = np.arange(male_survivor_percentage.count())
bar_width = 0.35

rect1 = ax.bar(index, male_survivor_percentage, bar_width, color='blue', label='Men')
rect2 = ax.bar(index + bar_width, female_survivor_percentage, bar_width, color='y', label='Women')

ax.set_ylabel('Survivor Percentage')
ax.set_title('Percentage Male and Female of survivors based on class')
xTickMarks = male_survivor_percentage.index.values.tolist()
ax.set_xticks(index + bar_width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

# What is the distribution of the non survivors among classes having family relative aboard the ship?

d['SibSp'].isnull().value_counts()
d['Parch'].isnull().value_counts()

non_survivors = d[(d['SibSp'] > 0) | (d['Parch'] > 0) & (d['Survived'] == 0)].groupby('Pclass')['Survived'].agg('count')
total_passengers = d.groupby('Pclass')['PassengerId'].count()
non_survivor_percentage = non_survivors / total_passengers

fig = plt.figure()
ax = fig.add_subplot(111)

rect = ax.bar(non_survivors.index.values.tolist(), non_survivors, color='blue', width=0.5)
ax.set_ylabel('No. of non survivors')
ax.set_title('Total number of non survivors with family based on class')
xTickMarks = non_survivors.index.values.tolist()
ax.set_xticks(non_survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

rect = ax.bar(non_survivor_percentage.index.values.tolist(), non_survivor_percentage, color='blue', width=0.5)
ax.set_ylabel('Non Survivor Percentage')
ax.set_title('Percentage of non survivors with family based on class')
xTickMarks = non_survivor_percentage.index.values.tolist()
ax.set_xticks(non_survivor_percentage.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()

#What was the survival percentage among different age groups?
d['Age'].isnull().value_counts()
age_bin = [0, 18, 25, 40, 60, 100]
d['AgeBin'] = cut(d.Age, bins=age_bin)
d_temp = d[np.isfinite(d['Age'])]  # removing all na instances

survivors = d_temp.groupby('AgeBin')['Survived'].agg(sum)
total_passengers = d_temp.groupby('AgeBin')['Survived'].agg('count')

pie(total_passengers, labels=total_passengers.index.values.tolist(),
    autopct='%1.1f%%', shadow=True, startangle=90)

title('Total Passengers in different age groups')
show()

pie(survivors, labels=survivors.index.values.tolist(),
    autopct='%1.1f%%', shadow=True, startangle=90)

title('Survivors in different age groups')

show()

