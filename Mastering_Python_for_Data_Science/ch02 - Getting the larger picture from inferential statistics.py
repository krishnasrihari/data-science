
%matplotlib inline
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)
x = [0, 1, 2, 3, 4, 5, 6]
n, p = 6, 0.5
rv = stats.binom(n, p)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
          label='Probablity')
ax.legend(loc='best', frameon=False)
plt.xlabel('No. of instances')
plt.ylabel('Probability')
plt.show()

fig, ax = plt.subplots(1, 1)
x = range(100)
n, p = 100, 0.5
rv = stats.binom(n, p)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
          label='Probablity')

ax.legend(loc='best', frameon=False)
plt.xlabel('No. of instances')
plt.ylabel('Probability')
plt.show()


fig, ax = plt.subplots(1, 1)
x = range(100)
n, p = 100, 0.4
rv = stats.binom(n, p)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
          label='Probablity')

ax.legend(loc='best', frameon=False)
plt.xlabel('No. of instances')
plt.ylabel('Probability')
plt.show()


fig, ax = plt.subplots(1, 1)
x = range(100)
n, p = 100, 0.6
rv = stats.binom(n, p)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
          label='Probablity')

ax.legend(loc='best', frameon=False)
plt.xlabel('No. of instances')
plt.ylabel('Probability')
plt.show()

fig, ax = plt.subplots(1, 1)
x = range(1000)
n, p = 1000, 0.5
rv = stats.binom(n, p)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
          label='Probablity')

ax.legend(loc='best', frameon=False)
plt.xlabel('No. of instances')
plt.ylabel('Probability')
plt.show()


rv = stats.poisson(20)
rv.pmf(23)

classscore = np.random.normal(50, 10, 60).round()

plt.hist(classscore, 20, normed=True)
plt.xlabel('No. of students')
plt.show()

stats.zscore(classscore)

prob = 1 - stats.norm.cdf(1.334)
prob

stats.norm.ppf(0.80)

(0.84 * classscore.std()) + classscore.mean()

zscore = ( 68 - classscore.mean() ) / classscore.std()
zscore

prob = 1 - stats.norm.cdf(zscore)
prob

height_data = np.array([186.0, 180.0, 195.0, 189.0, 191.0, 177.0, 161.0, 177.0, 192.0, 182.0, 185.0, 192.0,
                        173.0, 172.0, 191.0, 184.0, 193.0, 182.0, 190.0, 185.0, 181.0, 188.0, 179.0, 188.0,
                        170.0, 179.0, 180.0, 189.0, 188.0, 185.0, 170.0, 197.0, 187.0, 182.0, 173.0, 179.0,
                        184.0, 177.0, 190.0, 174.0, 203.0, 206.0, 173.0, 169.0, 178.0, 201.0, 198.0, 166.0,
                        171.0, 180.0])

plt.hist(height_data, 30, normed=True)
plt.xlabel('Height')
plt.show()

height_data.mean()

stats.sem(height_data)

average_height = []
for i in xrange(30):
    sample50 = np.random.normal(183, 10, 50).round()
    average_height.append(sample50.mean())

plt.hist(average_height, 10, normed=True)
plt.xlabel('Height')
plt.show()


average_height = []
for i in xrange(30):
    sample1000 = np.random.normal(183, 10, 1000).round()
    average_height.append(sample1000.mean())

plt.hist(average_height, 10, normed=True)
plt.xlabel('Height')
plt.show()


mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4,
       33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4]
hp = [110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245,
      175, 66, 91, 113, 264, 175, 335, 109]

stats.pearsonr(mpg, hp)

plt.scatter(mpg, hp)
plt.xlabel('mpg')
plt.ylabel('hp')
plt.show()


stats.pearsonr(mpg,hp)

stats.spearmanr(mpg,hp)

mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4,
       33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4, 120, 3]
hp = [110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245,
      175, 66, 91, 113, 264, 175, 335, 109, 30, 600]

plt.scatter(mpg, hp)
plt.xlabel('mpg')
plt.ylabel('hp')
plt.show()

stats.pearsonr(mpg, hp)

stats.spearmanr(mpg, hp)

class1_score = np.array([45.0, 40.0, 49.0, 52.0, 54.0, 64.0, 36.0, 41.0, 42.0, 34.0])
class2_score = np.array([75.0, 85.0, 53.0, 70.0, 72.0, 93.0, 61.0, 65.0, 65.0, 72.0])

stats.ttest_ind(class1_score, class2_score)


expected = np.array([6, 6, 6, 6, 6, 6])
observed = np.array([7, 5, 3, 9, 6, 6])

stats.chisquare(observed, expected)

men_women = np.array([[100, 120, 60],[350, 200, 90]])
stats.chi2_contingency(men_women)


country1 = np.array([ 176.,  179.,  180.,  188.,  187.,  184.,  171.,  201.,  172.,
        181.,  192.,  187.,  178.,  178.,  180.,  199.,  185.,  176.,
        207.,  177.,  160.,  174.,  176.,  192.,  189.,  187.,  183.,
        180.,  181.,  200.,  190.,  187.,  175.,  179.,  181.,  183.,
        171.,  181.,  190.,  186.,  185.,  188.,  201.,  192.,  188.,
        181.,  172.,  191.,  201.,  170.,  170.,  192.,  185.,  167.,
        178.,  179.,  167.,  183.,  200.,  185.])

country2 = np.array([ 177.,  165.,  175.,  172.,  179.,  192.,  169.,  185.,  187.,
        167.,  162.,  165.,  188.,  194.,  187.,  175.,  163.,  178.,
        197.,  172.,  175.,  185.,  176.,  171.,  172.,  186.,  168.,
        178.,  191.,  192.,  175.,  189.,  178.,  181.,  170.,  182.,
        166.,  189.,  196.,  192.,  189.,  171.,  185.,  198.,  181.,
        167.,  184.,  179.,  178.,  193.,  179.,  177.,  181.,  174.,
        171.,  184.,  156.,  180.,  181.,  187.])

country3 = np.array([ 191.,  190.,  191.,  185.,  190.,  184.,  173.,  175.,  200.,
        190.,  191.,  184.,  167.,  194.,  195.,  174.,  171.,  191.,
        174.,  177.,  182.,  184.,  176.,  180.,  181.,  186.,  179.,
        176.,  186.,  176.,  184.,  194.,  179.,  171.,  174.,  174.,
        182.,  198.,  180.,  178.,  200.,  200.,  174.,  202.,  176.,
        180.,  163.,  159.,  194.,  192.,  163.,  194.,  183.,  190.,
        186.,  178.,  182.,  174.,  178.,  182.])

stats.f_oneway(country1,country2,country3)
