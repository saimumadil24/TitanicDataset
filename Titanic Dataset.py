#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


#importing the titanic dataset
data=pd.read_csv(r'C:\Users\Saimum Adil Khan\OneDrive\Desktop\Python Course\train.csv',index_col='PassengerId')
test_data = pd.read_csv(r'C:\Users\Saimum Adil Khan\OneDrive\Desktop\Python Course\test.csv', index_col='PassengerId')
survived = pd.read_csv(r'C:\Users\Saimum Adil Khan\OneDrive\Desktop\Python Course\gender_submission.csv')


# In[3]:


#data view
data.head(20)


# In[4]:


#Checking Null Values
data.isnull().sum()


# In[5]:


#Filling null age cells with round figure
data['Age']=data['Age'].fillna(round(data['Age'].mean(),2))
data.head(20)


# In[6]:


#Data Formatting
data['Embarked']=data['Embarked'].replace({'S':'Southampton','C':'Cherbourg','Q':'Queenstown'})


# In[7]:


#Categorical Values into quantitative values
data['Sex']=data['Sex'].map({'male':0,'female':1})


# In[8]:


#binning
data['Age_Group']=pd.cut(data['Age'],bins=[0,12,18,60,100],labels=['Child','Teenager','Adult','Elderly'])


# In[9]:


#Categorical Values into quantitative values
data=pd.get_dummies(data,columns=['Embarked'],drop_first=True)


# In[10]:


#Removing unwanted columns
data=data.drop(['Name','Cabin','Ticket'],axis=1)


# In[11]:


data.head(10)


# In[12]:


#Selecting Only Numerical Values
numerical_data=data.select_dtypes(include=['int64','float64'])
numerical_data.head()


# In[13]:


#plt.figure(figsize=(8,14))
plt.boxplot(numerical_data)
plt.show()


# In[14]:


#checking error or not
err_fare=data[data['Fare']>400]
err_fare


# In[15]:


data.isnull().sum()


# In[16]:


#Normalization data


# In[17]:


data_main=data.copy()


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


scaler=StandardScaler().fit(data[['Age','Fare']])


# In[20]:


data[['Age','Fare']]=scaler.transform(data[['Age','Fare']])


# In[21]:


data.head(10)


# In[22]:


data_main.head(10)


# In[23]:


data.describe()


# In[24]:


data_main.describe()


# In[25]:


#Importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sb


# In[26]:


plt.subplot(1,2,1)
plt.hist(data['Age'])
plt.title('Age With Standardization')
plt.subplot(1,2,2)
plt.hist(data_main['Age'])
plt.title('Age Without Standardization')
plt.show()


# In[27]:


plt.subplot(1,2,1)
plt.hist(data['Fare'])
plt.title('Fare With Standardization')
plt.subplot(1,2,2)
plt.hist(data_main['Fare'])
plt.title('Fare Without Standardization')
plt.show()


# In[28]:


#making correlation
corr=data.corr()


# In[29]:


#Correlation by visualization
sb.heatmap(corr,annot=True)
plt.show()


# In[30]:


corr_main=data_main.corr()


# In[31]:


sb.heatmap(corr_main,annot=True)
plt.show()


# In[32]:


sb.pairplot(data)
plt.show()


# In[33]:


sb.pairplot(data_main)
plt.show()


# In[34]:


plt.scatter(data_main['Age'],data_main['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()


# In[35]:


#Logistic REgression and accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[36]:


X=data.drop(['Survived','Age_Group'],axis=1)
y=data['Survived']


# In[37]:


model=LogisticRegression()
model.fit(X,y)


# In[38]:


# Preprocess the test dataset
test_data['Age'] = test_data['Age'].fillna(round(test_data['Age'].mean(), 2))
test_data['Embarked'] = test_data['Embarked'].replace({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Age_Group'] = pd.cut(test_data['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teenager', 'Adult', 'Elderly'])
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)
test_data = test_data.drop(['Name', 'Cabin', 'Ticket'], axis=1)
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())

# Standardize the numerical variables
test_data[['Age', 'Fare']] = scaler.transform(test_data[['Age', 'Fare']])

# Make predictions using the trained logistic regression model
X_test = test_data.drop(['Age_Group'], axis=1)
y_pred = model.predict(X_test)

# Load the actual survival values
y_test = survived['Survived']

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')


# In[39]:


# Create a new DataFrame with the PassengerId and Survived columns
submission = pd.DataFrame({'PassengerId': test_data.index, 'Survived': y_pred})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

