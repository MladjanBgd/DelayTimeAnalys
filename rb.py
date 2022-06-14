# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:08:02 2022

@author: X
"""
#TODO
#[1] list of csv files
#[2]change plot type hist and make conclusion, it is Gaussian function?

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


print('Mladjan Jovanovic - Delay Time Analys ver 1.0')
#load data to pandas dataframe - df
##actualy we need to start from 2014 but there was issue with memory..
#df = pd.read_csv('D:/RB/2016.csv')
df = pd.concat(map(pd.read_csv, ['D:/RB/2016.csv', 'D:/RB/2017.csv','D:/RB/2018.csv']), ignore_index=True)

#print shape and info od pandas df
#print(df.shape)
#print(df.info())

#make delay mask
mask=df.columns.str.contains('.*DELAY')
#average delay time
ADT=df.loc[:,mask].mean()
#median delay time
MDT=df.loc[:,mask].median()
#skew
SDT=df.loc[:,mask].skew()
#kurt
KDT=df.loc[:,mask].kurt()



print('~~Mean delay time in minutes:')
print(ADT)
print('~~Median delay time in minutes:')
print(MDT)
print('~~~~~~~~~~~')
print('Total mean delay time in minutes: '+str(round(ADT.mean(),2)))
print('Total median delay time in minutes: '+str(round(MDT.mean(),2)))
print('\nLooking at the mean and median value we can conclude that the delay time does not have a symmetric bell shape.'
      '\nBecause mean is higer than the median we can conclude that we have major outliners with big delay time in minutes that skew delay time in +.'
      '\nThis is also proven by looking at the skew value'
      '\nOverall, in this year positive and negative delay times off all flights are annule i.e. ~0 min.')
print('~~~~~~~~~~~')
print('~~Skew delay time:')
print(SDT)
print('~~Kurt delay time:')
print(KDT)
print('~~~~~~~~~~~')
print('Total skew delay time: '+str(round(SDT.mean(),2)))
print('Total kurt delay time: '+str(round(KDT.mean(),2)))
print('\nLooking at the skew value (horizontal push) we can conclude that the distribution is moved to left'
      '\nand we have big kurt value (vertical push) so we have very big peak on that graph.')
print('~~~~~~~~~~~')



#https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
#plt=df.loc[:,mask].hist(bins=100)
        
#ploting histograms of delays
fig, ax = plt.subplots(3,3, figsize=(25,25))
ax=ax.ravel() #flaten 3x3 to 1x9
fig.suptitle('Histograms of Delay\'s')
ax[0].hist(df.loc[:,'DEP_DELAY'], color = 'blue', edgecolor = 'red', bins = 50)
ax[0].set_title('DEPARTURE DELAY')
ax[1].hist(df.loc[:,'ARR_DELAY'], color = 'blue', edgecolor = 'red', bins = 50)
ax[1].set_title('ARRIVAL DELAY')
ax[2].hist(df.loc[:,'CARRIER_DELAY'], color = 'blue', edgecolor = 'red', bins = 50)
ax[2].set_title('CARRIER DELAY')
ax[3].hist(df.loc[:,'WEATHER_DELAY'], color = 'blue', edgecolor = 'red', bins = 50)
ax[3].set_title('WEATHER DELAY')    
ax[4].hist(df.loc[:,'NAS_DELAY'], color = 'blue', edgecolor = 'red', bins = 50)
ax[4].set_title('NAS DELAY')  
ax[5].hist(df.loc[:,'SECURITY_DELAY'], color = 'blue', edgecolor = 'red', bins = 50)
ax[5].set_title('SECURITY DELAY')  
ax[6].hist(df.loc[:,'LATE_AIRCRAFT_DELAY'], color = 'blue', edgecolor = 'red', bins = 50)
ax[6].set_title('LATE AIRCRAFT DELAY')

for s in ax:
    s.set_xlabel('delay in min')
    s.set_ylabel('flights')

print('\nLooking at the histograms we can conlude that we have similar curve for all type of delays'
      '\nWe can represent Arrival Delay as normal distribution')    

plt.hist(df.loc[:,'ARR_DELAY'].dropna(), bins=75, density=True, alpha=0.7, color='blue')
mu, std = norm.fit(df.loc[:,'ARR_DELAY'].dropna()) #fix NaN
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  


 
#fix FL_DATE column as date format
df['FL_DATE']=pd.to_datetime(df.FL_DATE, format='%Y-%m-%d')
#and now calcualte average and median delay time   


DYER=df.groupby(df['FL_DATE'].dt.year)['ARR_DELAY'].agg(['mean', 'median'])
DQYR=df.groupby(df['FL_DATE'].dt.to_period('Q'))['ARR_DELAY'].agg(['mean', 'median'])
DMYR=df.groupby(df['FL_DATE'].dt.month)['ARR_DELAY'].agg(['mean', 'median'])
DWYR=df.groupby(df['FL_DATE'].dt.weekofyear)['ARR_DELAY'].agg(['mean', 'median'])
DDYR=df.groupby(df['FL_DATE'].dt.dayofweek)['ARR_DELAY'].agg(['mean', 'median'])
DCAR=df.groupby(df['OP_CARRIER'])['ARR_DELAY'].agg(['mean', 'median'])
DORG=df.groupby(df['ORIGIN'])['ARR_DELAY'].agg(['mean', 'median'])
DDES=df.groupby(df['DEST'])['ARR_DELAY'].agg(['mean', 'median'])

print(DYER,DQYR,DMYR,DWYR,DDYR,DCAR,DORG,DDES)

# for x in df['DEST']:
#     subset = df[df['DEST']==x]
#     plt.hist(subset['ARR_DELAY'])


print('~~~~~~~~~~~')
print('Finished.')

# #https://towardsdatascience.com/a-practical-guide-to-implementing-a-random-forest-classifier-in-python-979988d8a263
# import seaborn
# seaborn.pairplot(df.drop('ARR_DELAY', axis=1))
# seaborn.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns)

