#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
df = pd.read_csv('/blue/zoo6927/share/Jupyter_Content/data/auto_mpg.csv')


# In[6]:


df


# In[7]:


#two major factors affecting mpg is weight of the car and its engine displacement.
#mpg drops because more energy is needed to move more weight and a bigger displacement means more gas used

#I want to attemped to make a 3D graph because I don't think these factors are mutually exclusive
#older cars are heavier due to many things but mostly because they had bigger engines (read: more displacement)

import plotnine as pn
pn.ggplot(df, pn.aes(x='disp', y='weight')) + pn.geom_point()
# here is the relationship between weight and displacement


# In[8]:


pn.ggplot(df, pn.aes(x='disp', y='mpg', color='disp')) + pn.geom_point()
#relationship between displacement and miles per gallon


# In[9]:


pn.ggplot(df, pn.aes(x='weight', y='mpg', color='weight')) + pn.geom_point()
#relationship between weight and mpg


# In[10]:


from mpl_toolkits import mplot3d


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[30]:


#set up for 3D graph
fig = plt.figure().gca(projection='3d')

#relationship between weight, displacement, and mpg
fig.scatter(df.disp, df.weight, df.mpg,)

x = df['disp']
y = df['weight']
z = df['mpg']

ax.set_xlabel('disp')
ax.set_ylabel('weight')
ax.set_zlabel('mpg')

ax.scatter(x, y, z)

plt.show()

#how do i show a color gradient for 3D cause I cant see very well
#sagemath with Brian Stucky

# x = df[['disp', 'weight']]
# y = df.mpg

# model = LinearRegression(fit_intercept=True)

# model.fit(x, y)
# print('R^2:', model.score(x, y), 'MSE:', mse(model, x, y))

# print(model.intercept_, model.coef_)

#The Linear Regression plane that would show the relation of weight vs disp
#mpg = 43.8 + -0.0165*disp + -0.00575*weight


# In[23]:


pn.ggplot(df, pn.aes(x='weight', y='mpg', color='weight')) + pn.geom_pon()
#scatterplot of 


# In[14]:


from sklearn.linear_model import LinearRegression

#define terms 
x=df[['weight']]
y=df.mpg

model = LinearRegression()
model.fit(x, y)


# In[15]:


model = LinearRegression().fit(x,y)


# In[16]:


#print intercept, coefficient, and R^2 value
print(model.intercept_, model.coef_)
model.score(x, y)


# In[17]:


(pn.ggplot(df, pn.aes(x='weight', y='mpg', color="weight")) + pn.geom_point() 
 + pn.geom_abline(intercept=model.intercept_, slope=model.coef_[0], color='red'))


# In[18]:


#for multiple linear regression
df['weight_sq'] = df.weight**2

x2 = df[['weight', 'weight_sq']]
x2


# In[19]:


#print intercept, coefficient, and R^2 value

model = LinearRegression().fit(x2, y)
print(model.intercept_, model.coef_)
model.score(x2, y)


# In[20]:


#makeing a new dataframe seems easier to me because I feel like it removes chances of errors

xvals = np.linspace(x2.weight.min(), x2.weight.max(), len(df))
xy_predict = pd.DataFrame({
    'weight': xvals,
    'weight_sq':xvals**2
})

xy_predict['y']=model.predict(xy_predict)

(pn.ggplot(df, pn.aes(x='weight', y='mpg', color='weight')) + pn.geom_point() + 
    pn.geom_line(data=xy_predict, mapping=pn.aes(x='weight', y='y'), color='red')
)


# In[ ]:




