#!/usr/bin/env python
# coding: utf-8

# ### Building a classification model for titanic data for survival or non survival 

# In[2]:


import pandas as pd
import seaborn as sns


# In[3]:


# reading titanic data set
df =sns.load_dataset('titanic')


# In[4]:


df


# In[5]:


# sns.barplot(data = df,x='sex',y='tip')


# In[6]:


# percentages of missing values in each columns

df.isnull().sum()/df.shape[0]*100


# In[7]:


#droping deck column from the dataset because it is so related for survival or non survival prediction

df.drop(columns=['deck'],inplace=True)


# In[8]:


# Now we plot a graph which show that on the basis of embarked column the average age of people 
# from each city

sns.catplot(data = df,x='embarked',y='age',kind='bar')


# In[10]:


# the below graph show that male survival on the basis of city on first row
# on the next row the survival of female on the basis of city 
sns.displot(data = df,x='age',hue='survived',col='embarked',row='sex',kind='kde')


# In[179]:


# here i have plot histogram plot of survival of male and female on basis of city
sns.displot(data = df,x='age',hue='survived',col='embarked',row='sex',kind='hist',element='step')


# In[180]:


# to check how many missing values in each columns

df.isnull().sum()


# In[181]:


# here we find all those rows in age columns where the value is null 
age_null = df[df['age'].isnull()]


# In[182]:


age_null


# In[183]:


df['age'].mean()


# In[184]:


df['age'].median()


# In[185]:


# counting people that come from the three city
age_null['embarked'].value_counts()


# In[186]:


# counting the male and female 
age_null['sex'].value_counts()


# In[187]:


# counting the number of people in three category seats

age_null['pclass'].value_counts()


# In[188]:


# the people who are male , and they are in 3rd category seat and come from southampton(S)
df[(df['sex']=='male') &(df['pclass']==3) &(df['embarked']=='S')].shape


# In[189]:


# age missing value filling 


# In[190]:


# filling all missing values in age column with mean of age
df['age'].fillna(df['age'].mean(),inplace=True)


# In[191]:


df.info()


# In[192]:


df


# In[ ]:





# In[195]:


# ploting bar graph on basis of male and female
sns.catplot(data = df ,x='sex',y='age',kind='bar')


# In[ ]:





# In[196]:


# on the basis of fare column ploting histogram
sns.displot(data = age_null,x='fare',hue='survived',col='embarked',row='sex',kind='hist',element='step')


# In[197]:


df.drop(columns=['alive','adult_male','embark_town'],inplace=True)


# In[198]:


df.drop(columns=['who'],inplace=True)


# In[199]:


df['parch'].value_counts()


# In[200]:


df.drop(columns=['parch'],inplace=True)


# In[201]:


df.drop(columns=['alone'],inplace=True)


# In[202]:


df


# In[203]:


df.info()


# In[204]:


df['embarked'].mode()


# In[205]:


df['embarked'].fillna(df['embarked'].mode(),inplace=True)


# In[206]:


df


# In[207]:


df.drop(columns='class',inplace=True)


# In[208]:


df


# In[209]:


df.isnull().sum()


# In[210]:


df['embarked'].mode()[0]


# In[211]:


df['embarked'].fillna(df['embarked'].mode()[0],inplace=True)


# In[212]:


df.isnull().sum()


# In[213]:


# not data will be clean


# In[214]:


sns.boxplot(data=df)


# In[215]:


sns.boxplot(data =df,y='fare')


# In[216]:


q1 = df['fare'].quantile(0.25)
q1


# In[217]:


q3 = df['fare'].quantile(0.75)
q3


# In[218]:


iqr = q3-q1


# In[219]:


iqr


# In[220]:


min = q1 -1.5 *iqr
min


# In[221]:


max = q3 + 1.5 *iqr
max


# In[222]:


df=df[((df['fare']>-26.724 )& (df['fare']<65.6344))]


# In[223]:


df


# In[224]:


from sklearn.preprocessing import LabelEncoder


# In[225]:



# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the column
df['embarked'] = label_encoder.fit_transform(df['embarked'])


# In[226]:


df


# In[227]:


# Fit and transform the column
df['sex'] = label_encoder.fit_transform(df['sex'])


# In[228]:


df


# In[229]:


x=df.drop('survived',axis=1)
y=df['survived']


# In[240]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=.20,random_state=101)


# In[241]:


from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()


# In[242]:


lreg.fit(x_train, y_train) 


# In[243]:


y_pred_lreg = lreg.predict(x_test)


# In[244]:


y_pred_lreg


# In[249]:


probabilities = lreg.predict_proba(x_test)[:,1]
probabilities


# In[254]:


for i in range(len(y_test)):
    print(round(probabilities[i],3),'--->',y_pred_lreg[i])


# In[245]:


from sklearn.metrics import classification_report,accuracy_score

print('Classification Model')

# accuracy
print('--'*30)
logreg_accuracy = round(accuracy_score(y_test,y_pred_lreg)*100,2)
print('Accuracy',logreg_accuracy,'%')


# In[ ]:




