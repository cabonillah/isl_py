import os
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("D:/Archivos/Datos_practica/Python")


with ZipFile('../ISLR/ALL+CSV+FILES+-+2nd+Edition+-+corrected.zip') as zf:
    college = pd.read_csv(zf.open('ALL CSV FILES - 2nd Edition/College.csv'))

college.set_axis(college.iloc[:, 0], axis='index', inplace=True)
college.drop(college.columns[0], axis=1, inplace=True)
college['Private'] = np.where(college.Private == 'Yes', 1, 0)
college['Private'] = college.Private.astype('category')
college.describe()

pd.plotting.scatter_matrix(college)
plt.close()

plt.figure()
sns.boxplot(x=college['Private'], y=college['Outstate'])
plt.close()

college['Elite'] = np.where(college['Top10perc'] > 50, 1, 0)
college['Elite'] = college['Elite'].astype('category')
college['Elite'].describe()

plt.figure()
sns.boxplot(x=college['Elite'], y=college['Outstate'])
plt.close()

with ZipFile('../ISLR/ALL+CSV+FILES+-+2nd+Edition+-+corrected.zip') as zf:
    auto = pd.read_csv(zf.open('ALL CSV FILES - 2nd Edition/Auto.csv'), 
                       na_values=['?'])

auto.dropna(inplace=True)
auto.info()
auto.head()

subset1 = auto.drop(auto.columns[[7, 8]], axis = 1)

subset1.max(axis=0)-subset1.min(axis=0)
subset1.mean(axis=0)
subset1.std(axis=0)

subset2 = subset1.drop(subset1.loc[11:87].index)

subset2.max(axis=0)-subset2.min(axis=0)
subset2.mean(axis=0)
subset2.std(axis=0)

plt.figure()
pd.plotting.scatter_matrix(auto)
plt.close()

data_url_boston = "http://lib.stat.cmu.edu/datasets/boston"
raw_df_boston = pd.read_csv(data_url_boston, sep="\s+", skiprows=22, header=None)
data_boston = np.hstack([raw_df_boston.values[::2, :], raw_df_boston.values[1::2, :2]])
target_boston = raw_df_boston.values[1::2, 2]


Boston = pd.DataFrame(data_boston)
medv = pd.DataFrame(target_boston)
Boston = pd.concat([Boston, medv], axis=1)
Boston.columns = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis",
                  "rad", "tax", "ptratio", "black", "lstat", "medv"]


plt.figure()
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(Boston['crim'], Boston['ptratio'], marker='1', color='#DB1BF6')
axs[0, 1].scatter(Boston['crim'], Boston['rm'], marker='*', color='#8917D4')
axs[1, 0].scatter(Boston['crim'], Boston['age'], marker='p', color='#6726EB')
axs[1, 1].scatter(Boston['crim'], Boston['ptratio'], marker='h', color='#1B48F6')
plt.close()

variables = ['crim', 'tax', 'ptratio']

for i in variables:
    plt.figure()
    plt.hist(Boston[i])
    plt.xlabel(f'{i}')
    plt.ylabel('count')
    plt.close()

summary_stats = Boston.loc[:, variables].aggregate(func=['mean','std'], axis=0)

high_stats = []

for i in range(3):
    high_stat =  summary_stats.iloc[0, i] + 1.5 * summary_stats.iloc[1, i]
    high_stats.append(high_stat)

high_stats = pd.DataFrame(high_stats).T
high_stats.columns = ['crim_high', 'tax_high', 'ptratio_high']

Boston.loc[Boston['crim'] > high_stats.loc[0, 'crim_high']]
Boston.loc[Boston['tax'] > high_stats.loc[0, 'tax_high']]
Boston.loc[Boston['ptratio'] > high_stats.loc[0, 'ptratio_high']]

Boston['chas'].sum()

Boston[Boston['medv'] == Boston['medv'].min()]
summary_stats
high_stats

Boston.query('rm > 7').filter(['crim', 'tax', 'ptratio']).aggregate('mean')
Boston.query('rm > 8').filter(['crim', 'tax', 'ptratio']).aggregate('mean')
summary_stats
high_stats
