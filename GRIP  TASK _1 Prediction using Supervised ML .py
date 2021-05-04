#!/usr/bin/env python
# coding: utf-8

# # TASK_1 Prediction using Supervised ML
# 
# 
# ## THE SPARK FOUNDATION
# ### GRIPMAY21
# ### INTERN - SACHPREET SINGH
# 
# 
# #### Predict the percentage of an student based on the no. of study hours.

# In[1]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[4]:


# Reading data from remote link

db = pd.read_csv("http://bit.ly/w-data")
print("Data imported successfully")
db.head(25)


# In[5]:


#visualizing the data
db.plot(x = 'Hours', y = 'Scores', style = 'o')
plt.title('Hours vs Scores', size = 15)
plt.xlabel('Hours Studied', size = 12)
plt.ylabel('Percentage Score', size = 12)
plt.show()


# In[6]:


#dividing the data into 'attributes' and 'labels'
x = db.iloc[:, :-1].values
y = db.iloc[:, 1].values

#spliting the data into training and test sets.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[8]:


#training the model
regression = LinearRegression()
regression.fit(x_train, y_train)

print("Model Trained")


# In[9]:


sns.regplot(x = db['Hours'], y = db['Scores'])
plt.title('Regression Plot', size = 15)
plt.xlabel('Hours Studied', size = 12)
plt.ylabel('Percentage Score', size = 12)
plt.show()


# In[10]:


#now predicting the percentage score
y_pred = regression.predict(x_test)

#comparing the predicted score with actual score
prediction = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
prediction


# In[11]:


#visualizing predicted score vs actual score
plt.scatter(x = x_test, y = y_test, color = 'blue', label = 'Actual')
plt.plot(x_test, y_pred, color = 'red', label = 'Predicted')
plt.legend()
plt.title('Actual vs Predicted', size=15)
plt.ylabel('Percentage Score', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[12]:


#evaluating the model, calculating the accuracy of the model
print('Mean absolute error: ', mean_absolute_error(y_test, y_pred))


# In[16]:


#What will be predicted score if a student studies for 9.25 hrs/ day?
hours = [9.25]cccc
ans = regression.predict([hours])
print("Score = {}".format(round(ans[0],3)))


# 
# According to the regression model if a student studies for 9.25 hours a day  is likely to score 93.692 marks.
# 
