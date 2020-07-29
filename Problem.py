import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#reading data & columns
data=pd.read_csv('D:/IP_LP3_DATA_SCIENCE_DEBJYOTI_SAHA_2982/Week 1/DS_DATESET.csv')
print(data)
col=data.columns
print(col)


#Question a
c=data['Major/Area of Study'].value_counts()
print(c)
intr=data['Areas'].value_counts()
print(intr)
intr1=data['Areas'].value_counts().plot.bar(color='g')
plt.show()


#Question b
Lang= data['Programming Language Known other than Java (one major)'].value_counts()
print(Lang)
Lang1= Lang.plot.bar(color='g')
plt.show()


#Question c
learn= data['How Did You Hear About This Internship?'].value_counts()
print(learn)
learn1= learn.plot.bar(color='g')
plt.show()


#Question d
year= data['Which-year are you studying in?'].value_counts()
print(year)
year1= year.plot.bar(color='g')
plt.show()


#Question e
comm=data['Rate your written communication skills [1-10]'].value_counts()
print(comm)
comm1=comm.plot.bar(color='g')
plt.show()

verbal=data['Rate your verbal communication skills [1-10]'].value_counts()
print(verbal)
verbal1=verbal.plot.bar(color='g')
plt.show()


#Question f
year= data['Which-year are you studying in?'].value_counts()
print(year)
year1= year.plot.bar(color='g')
plt.show()

intr=data['Areas'].value_counts()
print(intr)
intr1=data['Areas'].value_counts().plot.bar(color='g')
plt.show()


#Question g
city= data['City'].value_counts()
print(city)
city1=city.plot.bar(color='g')
plt.show()

college=data['College name'].value_counts()
print(college)
college1=college.plot.bar(color='g')
plt.show()


#Question h
cgpa=data['CGPA/ percentage']
target=data['Label']
print(target)
ypos=np.arange(len(cgpa))
print(ypos)
plt.xticks(ypos,cgpa)
plt.scatter(ypos,cgpa, label='CGPA/ percentage')
plt.show()


#Question i
interest=data['Areas']
target=data['Label']
print(target)
ypos=np.arange(len(interest))
print(ypos)
plt.xticks(ypos,interest)
plt.bar(ypos,interest, label='Areas')
plt.show()


#Question j
year2=data['Which-year are you studying in?']
major=data['Major/Area of Study']
target=data['Label']
ypos=np.arange(len(year2))
print(ypos)
plt.xticks(ypos,year2)
plt.scatter(ypos,major,label='Major/Area of Study')
plt.show()
plt.bar(ypos,target, label='Label')
plt.show()




