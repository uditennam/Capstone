#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:11:53 2018

@author: uditennam
"""
## importing libraries
import numpy as np
import pandas as pd
import json
# customer will spend money or not
# how much money the customer will spend

train = pd.read_csv('train.csv',dtype = {"fullVisitorId":str})

train.head()
# conversion from json objects to dict and transforming them into columns
totals = train['totals'].apply(lambda x:json.loads(x))
totals_head = totals.apply(pd.Series)
new_names = [(i,'totals_'+i) for i in totals_head.columns.values]
totals_head.rename(columns = dict(new_names), inplace = True)

device = train['device'].apply(lambda x:json.loads(x))
device_head = device.apply(pd.Series)
new_names = [(i,'device_'+i) for i in device_head.columns.values]
device_head.rename(columns = dict(new_names), inplace = True)

geoNetwork = train['geoNetwork'].apply(lambda x:json.loads(x))
geoNetwork_head = geoNetwork.apply(pd.Series)
new_names = [(i,'geoNetwork_'+i) for i in geoNetwork_head.columns.values]
geoNetwork_head.rename(columns = dict(new_names), inplace = True)

trafficSource = train['trafficSource'].apply(lambda x:json.loads(x))
trafficSource_head = trafficSource.apply(pd.Series)
new_names = [(i,'trafficSource_'+i) for i in trafficSource_head.columns.values]
trafficSource_head.rename(columns = dict(new_names), inplace = True)

train.drop(['device','geoNetwork','totals','trafficSource'], axis = 1, inplace = True)
df = pd.concat([train,device_head,geoNetwork_head,totals_head,trafficSource_head], axis = 1)

len(df.columns) # 50 columns

df.info()
len(df['fullVisitorId'].unique())

len(df)
dictionary = {}
for i in df.columns.values:
    val = df[i].isnull().sum()
    percent = str(round((val/903653)*100,2))
    dictionary[i] = percent
    
#dictionary
df_null = pd.DataFrame.from_dict(dictionary, orient = 'index')
df_null.columns = ['Percentage_NullEntries']
#df_null
#df_null.info()
df_null.reset_index(level = 0, inplace = True)
df_null.columns = ['ColumnName','Percent_NullEntries']
df_null
df_null['Percent_NullEntries']=df_null['Percent_NullEntries'].astype(float)
major_nulls = df_null[df_null['Percent_NullEntries']>50]

df_Null = df[df['totals_transactionRevenue'].isnull()]
dictionary_Null = {}
for i in df_Null.columns.values:
    val = df_Null[i].isnull().sum()
    percent = str(round((val/892138)*100,2))
    dictionary_Null[i] = percent
    
df_Null_null = pd.DataFrame.from_dict(dictionary_Null, orient = 'index')
df_Null_null.columns = ['Percentage_NullEntries']
#df_null
#df_null.info()
df_Null_null.reset_index(level = 0, inplace = True)
df_Null_null.columns = ['ColumnName','Percent_NullEntries']
df_Null_null
df_Null_null['Percent_NullEntries']=df_Null_null['Percent_NullEntries'].astype(float)
major_Null_nulls = df_Null_null[df_Null_null['Percent_NullEntries']>50]
major_Null_nulls
df_Null.drop(['trafficSource_adContent','trafficSource_campaignCode','trafficSource_keyword',
              'trafficSource_referralPath'], axis = 1, inplace = True)
    
df_Null['trafficSource_isTrueDirect'].value_counts()    
df_Null['trafficSource_isTrueDirect']=df_Null['trafficSource_isTrueDirect'].replace(np.nan, False)
df_Null.head()
df_Null.reset_index(drop = True, inplace = True)
df_Null['totals_transactionRevenue'] = df_Null['totals_transactionRevenue'].replace(np.nan, 0)

df_Null_null[df_Null_null['Percent_NullEntries']<50]
# totals_bounces , totals_new visits have < 50% null entries
df_nonNull.columns
df_nonNull.columns
df_Null['totals_bounces'].value_counts() # other times it is 0
df_Null['totals_bounces']=df_Null['totals_bounces'].replace(np.nan, 0)
df_Null['totals_newVisits'].value_counts() # new visit 1 , old visit 0 it is like boolean
df_Null['totals_newVisits']=df_Null['totals_newVisits'].replace(np.nan, 0)

df_Null.drop('date', axis = 1, inplace = True)

len(df_Null.columns) # 45 columns - additional columns are totals_bounces, totals_newVisits
len(df_nonNull.columns) # 43 columns

# AKHILESH - Null Data
# UDIT - Non Null Data

df_nonNull.columns[df_nonNull.isnull().any()].tolist() # no null entries here
df_Null.columns[df_Null.isnull().any()].tolist()
df_Null['totals_pageviews'].value_counts()
df_Null['totals_newVisits'].value_counts()

df_Null['totals_pageviews'].median()
sns.jointplot(x = 'totals_newVisits', y = 'totals_pageviews', data = df_Null)
df_Null.groupby('socialEngagementType', as_index=False)['totals_pageviews'].mean()
df_nonNull['totals_pageviews']=df_nonNull['totals_pageviews'].astype('float')
df_nonNull.groupby('socialEngagementType', as_index=False)['totals_pageviews'].mean()
df['socialEngagementType'].value_counts()
df_Null.drop('socialEngagementType', axis = 1, inplace = True)
df_nonNull.drop('socialEngagementType', axis = 1, inplace = True)

df_Null.groupby('visitNumber', as_index=False)['totals_pageviews'].mean()

df_Null.dtypes
df_Null['totals_newVisits']=df_Null['totals_newVisits'].astype('int')
df_Null['totals_pageviews'] = df_Null['totals_pageviews'].astype('float')
df_Null['totals_pageviews'] = df_Null['totals_pageviews'].replace(np.nan, df_Null['totals_pageviews'].median())
df_Null['totals_pageviews'].mean() # for normal distribution, you can use mean

df_Null.columns[df_Null.isnull().any()].tolist()
df_Null.to_csv()
df_Null.reset_index(drop = True, inplace = True)

df_Null

df_nonNull

# convert df_Null and df_nonNull to csv formats and upload on bitbucket

df_nonNull['trafficSource_adwordsClickInfo']

df_nonNull.loc[:, (df_nonNull != df_nonNull.iloc[0]).any()]

df_nonNull.dtypes


df_nonNull_tsa = df_nonNull[['visitStartTime','totals_transactionRevenue']]

df_nonNull.drop('visitStartTime', axis = 1, inplace = True)

# removing columns with constant values
df_nn = df_nonNull.loc[:, (df_nonNull != df_nonNull.iloc[0]).any()]

df_Null_tr = df_Null[['totals_transactionRevenue']]
df_n = df_Null.loc[:, (df_Null != df_Null.iloc[0]).any()]

# add back the tr column to the df
df_n = pd.concat([df_n, df_Null_tr], axis = 1)



set(df_n.columns).difference(set(df_nn.columns))

# convert visitStartTime 

df_nn_tsa = df_nn[['visitStartTime','totals_transactionRevenue']]
df_n_tsa = df_n[['visitStartTime','totals_transactionRevenue']]
df_nn_tsa


# 4 dataframes to work with
# df_nn, df_n, df_nn_tsa, df_n_tsa

# convert these 4 dataframes into csv files
df_n_tsa.to_csv('df_Null_TSA', index = False)
df_nn_tsa.to_csv('df_nonNull_TSA', index = False)

df_n.to_csv('df_Null', index = False)
df_nn.to_csv('df_nonNull', index = False)



df_Null['totals_pageviews'].value_counts()
## Check for Columns with constant values
const_cols = [c for c in df_Null.columns if df_Null[c].nunique(dropna=False)==1 ]


const_cols

set(df_Null.columns).difference(set(df_nonNull.columns))
######################################
# splitting dataframe into 2 parts - non null rows of transaction revenue and null entries
# first we check for the non-null entries
df_nonNull = df[~df['totals_transactionRevenue'].isnull()] 
df_nonNull.reset_index(drop = True, inplace = True)
len(df_nonNull) # 11515 rows
dictionary_nonNull = {}
for i in df_nonNull.columns.values:
    val = df_nonNull[i].isnull().sum()
    percent = str(round((val/11515)*100,2))
    dictionary_nonNull[i] = percent
    
df_nonNull_null = pd.DataFrame.from_dict(dictionary_nonNull, orient = 'index')
df_nonNull_null.columns = ['Percentage_NullEntries']
#df_null
#df_null.info()
df_nonNull_null.reset_index(level = 0, inplace = True)
df_nonNull_null.columns = ['ColumnName','Percent_NullEntries']
df_nonNull_null
df_nonNull_null['Percent_NullEntries']=df_nonNull_null['Percent_NullEntries'].astype(float)
major_nonNull_nulls = df_nonNull_null[df_nonNull_null['Percent_NullEntries']>50]
major_nonNull_nulls    

#df_nonNull
# Dropping few columns
df_nonNull = df_nonNull.drop(['totals_bounces','totals_newVisits','trafficSource_adContent',
                              'trafficSource_campaignCode','trafficSource_keyword',
                              'trafficSource_referralPath'], axis = 1)
    
df_nonNull.head()

df_nonNull['totals_transactionRevenue'].value_counts()
df_nonNull['fullVisitorId'].value_counts()
df_nonNull['titals_transactionRevenue'].mean() # infinite
df_nonNull['totals_transactionRevenue'].median() # 49450000.0
########################################

# filling < 50% null entries
df_nonNull_null
df_nonNull['trafficSource_isTrueDirect'].value_counts()

df_nonNull['trafficSource_isTrueDirect'] = df_nonNull['trafficSource_isTrueDirect'].replace(np.nan,False)


## ANALYSIS DF_NONULL DATAFRAME ##
# checking for any null columns
df_nonNull.columns[df_nonNull.isnull().any()].tolist()
# all are non-null columns
# lets' dive into analysis
df_nonNull.columns

num_cols = df_nonNull._get_numeric_data().columns # Numeric columns
category_cols = list(set(df_nonNull.columns) - set(num_cols))

len(num_cols) # No of numeric cols = 6
len(category_cols) # No of categorical cols = 38

# Data needs some processing still 
# starting with num_cols

num_cols
# date
# visitId
# visitNumber
# visitStartTime
# device_isMobile
# trafficSource_isTrueDirect
df_nonNull['date']
df_nonNull['visitNumber'] # no of visits
df_nonNull['visitStartTime'] # posix or unix time
df_nonNull['device_isMobile'].value_counts() # can be converted to nominal categorical var - T and F
df_nonNull['trafficSource_isTrueDirect'].value_counts() # can be converted to nominal categorical var - T and F
df_nonNull['visitId'] # a very long int

## Conversion to datetime format
from datetime import datetime
df_nonNull['date']=df_nonNull['date'].astype('str')
df_nonNull['date_year'] = df_nonNull['date'].apply(lambda x:x[:4])
df_nonNull['date_month'] = df_nonNull['date'].apply(lambda x:x[4:6])
df_nonNull['date_date'] = df_nonNull['date'].apply(lambda x:x[6:])

df_nonNull['date']=df_nonNull['date_year'] + '-' + df_nonNull['date_month'] + '-' + df_nonNull['date_date']

import time
df_nonNull['visitStartTime']=df_nonNull['visitStartTime'].apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))

df_nonNull['visitStartTime'].

from datetime import datetime
df_nonNull['visitStartTime'] = df_nonNull['visitStartTime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

num_cols = df_nonNull._get_numeric_data().columns # Numeric columns
category_cols = list(set(df_nonNull.columns) - set(num_cols))

len(num_cols) # no of numeric cols = 4
len(category_cols) # no of category cols = 43

df_nonNull.dtypes
# dropping a few more cols
df_nonNull.drop(['date','date_year','date_month','date_date'], axis = 1, inplace = True)

num_cols = df_nonNull._get_numeric_data().columns # Numeric columns
category_cols = list(set(df_nonNull.columns) - set(num_cols))

len(num_cols) # 4
len(category_cols) # 39

df_nonNull.dtypes

# Univariate analysis
# finding out the categorical(boolean can be considered as categorical too) and numerical variables

# NUMERICAL COLUMNS

num_cols
# visitId, visitNumber, device_isMobile, trafficSource_isTrueDirect
# out of the 2 num_cols, 2 are boolean
# COUNT - BAR || COUNT % - PIE
# visitId can or cannot be considered as a feature
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))
sns.countplot(df_nonNull['visitNumber'])
plt.xticks(rotation = 90)

plt.hist(df_nonNull.visitNumber, bins = 50)


###################################################################################
# time series analysis can be done on visitStartTime and totals_transactionRevenue

df_tsa = df_nonNull[['visitStartTime','totals_transactionRevenue']]
# created a new dataframe for Time series analysis

# perform TSA here










###################################################################################













#########################################





















import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x = 'ColumnName', y = 'Percent_NullEntries', data = major_nulls)
plt.xticks(rotation = 90)

df['channelGrouping'].value_counts()


#df_null = pd.DataFrame(dictionary.items(),columns = ['Column_Name','Percentage_NullEntries'])

dictionary
