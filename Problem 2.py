import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#Data Pre-processing
data=pd.read_csv('D:/IP_LP3_DATA_SCIENCE_DEBJYOTI_SAHA_2982/Week 1/DS_DATESET.csv')
print(data)
col=data.columns
print(col)

#null value counts
n_ull=data.isna().sum()
print(n_ull)

#drop unnecessary columns
eli=data.drop(['First Name', 'Last Name', 'Zip Code','Age', 'Degree'], axis=1, inplace=True)
print(data.head(10))

#total missing values
total=data.isnull().sum().sort_values(ascending=False)
print(total)
t_head=total.head()
print(t_head)

#percentage of missing value
percent=(data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
print('The percent value is')
print(percent)
missing_data=pd.concat([total,percent],axis=1,keys=['Total', 'Percent'])
print(missing_data)
miss_head=missing_data.head()
print(miss_head)

#percentage of missing value(bar-plot)
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=20)
plt.ylabel('Percent of missing values', fontsize=20)
plt.title('Percent of missing values by feature', fontsize=20)
plt.show()

