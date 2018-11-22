
# coding: utf-8

# In[982]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import random


# In[983]:


#Importing Data
os.chdir("D:/Project")#set working directory
os.getcwd()#get current working directory
df=pd.read_csv('day.csv', sep=',', parse_dates=True, index_col="dteday")#read data with "dteday" as index


# In[984]:


#EDA
df.head(5)#Looking at first 5 rows of data
df=df.drop('instant', axis=1)#droping "instant" as it does not contain any information


# In[985]:


df.info()#getting more information


# In[986]:


cat_list=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday','weathersit']#Identifieng categorical data
num_list=['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']#Identifieng numerical data


# In[987]:


#Visual EDA
#Bivariate analysis of categorical data
for i in cat_list:
    plt.figure(figsize=(15,5))
    df.groupby(i)['cnt'].sum().plot(kind='bar')
    plt.show()


# In[988]:


#missing value
assert df.notnull().all().all()


# In[989]:


#Outlier Analysis by using boxplot
for i in num_list:
    plt.figure(figsize=(5,5))
    plt.boxplot(df[i])
    plt.xlabel(i)
    plt.show()


# In[990]:


#Identifing outliers using numerical method and dropping them
for i in num_list:
    q75, q25= np.percentile(df.loc[:,i], [75, 25])
    iqr=q75-q25
    min=q25 - (iqr*1.5)
    max=q75 + (iqr*1.5)
    df.loc[df[i]<min,:]
    df.loc[df[i]>max,:]
    df = df.drop(df[df.loc[:,i] < min].index)
    df = df.drop(df[df.loc[:,i] > max].index)


# In[991]:


#temp, atemp, hum and windspeed have already been normalized
#normalizing target variables
new_list=['registered', 'casual', 'cnt']
for i in new_list:
    df[i]= (df[i]-np.min(df[i]))/(np.max(df[i])-np.min(df[i]))
df.head()


# In[993]:


#Featuere selction
f, ax = plt.subplots(figsize=(15, 8))
corr_matrix = df.loc[:, num_list].corr()

#Plot using seaborn library
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), 
           cmap=sns.diverging_palette(220, 50, as_cmap=True),
            square=True, ax=ax, annot = True)
plt.plot()


# In[994]:


#loop for ANOVA test Since the target variable is continuous
for i in cat_list:
    f, p = stats.f_oneway(df[i], df["season"])
    print("P value for variable "+str(i)+" is "+str(p))


# In[995]:


#Dropping correlated variables and target variables and updating categorical and numerical list
df=df.drop(['registered', 'temp', 'casual'], axis=1)
cat_list=['mnth', 'yr', 'holiday', 'weekday', 'workingday','weathersit', 'season']
num_list=['atemp', 'hum', 'windspeed']


# In[996]:


#getting dummies
df_cat = pd.get_dummies(data = df, columns = cat_list)
df1=df_cat.copy()


# In[997]:


#X_train, X_test, y_train, y_test = train_test_split( df_cat.iloc[:, df_cat.columns != 'cnt'], df_cat['cnt'], test_size = 0.20)


# In[998]:


#model1 = DecisionTreeRegressor(max_depth = 3,  min_samples_leaf = 0.12, random_state=42)
#model1.fit(X_train, y_train)
#pred1 = model1.predict(X_test)
#print(np.sqrt(mean_squared_error(y_test,pred1)))
#RMSE: 0.14159746411572258


# In[999]:


#model2 = LinearRegression()#Initiate model
#model2.fit(X_train , y_train)#fit on training data
#pred2 = model2.predict(X_test)#predict on test data
#print(np.sqrt(mean_squared_error(y_test, pred2)))#Root mean square eror
#RMSE: 0.10932442082291012


# In[1000]:


#model3 = RandomForestRegressor()#Intiate model
#model3.fit(X_train, y_train)#fit on training ata
#pred3 = model3.predict(X_test)#predict on test data
#print(np.sqrt(mean_squared_error(y_test, pred3)))#Root mean square eror
#RMSE: 0.09530872087446146


# In[1001]:


target = df_cat['cnt']
df_cat.drop(['cnt'], inplace = True, axis=1)
print(df_cat.shape)
features=df1


# In[1002]:


#splitting data into train and test
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.20, random_state=129)


# In[1003]:


#PCA
pca = PCA()#calling an instance of PCA 
pca.fit(features_train)

# The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.show()


# In[1004]:


#As per graph selecting only 20 features
pca = PCA(n_components=20)#As per graph selecting only 20 features
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)


# In[1005]:


#Decision Tree
model_dt= DecisionTreeRegressor(random_state=1234)#initiate model
model_dt.fit(features_train, target_train)#fit to train data
pred_dt= model_dt.predict(features_test)#predict on test data


# In[1006]:


#Hyperparameter Tuning and cross validation
parameters_dt = {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10]}
param_dt = GridSearchCV(model_dt, parameters_dt, cv=5)
param_dt.fit(features_train, target_train)
model_dt_best = param_dt.best_estimator_
pred_dt_best=model_dt_best.predict(features_test)
print(param_dt.best_estimator_)
rmse_dt_best =np.sqrt(mean_squared_error(target_test,pred_dt_best))
mape_dt_best = np.mean((np.abs(target_test - pred_dt_best) / target_test))*100
print("MAPE for Decision Tree= "+str(mape_dt_best))
print("Root Mean Squared Error For Decision Tree= "+str(rmse_dt_best))
print("R^2 Score For Decision Tree= "+str(r2_score(target_test,pred_dt_best)))


# In[1007]:


#Root Mean Squared Error For Decision Tree= 0.11456320372193872
#R^2 Score For Decision Tree= 0.7248744303357981
#MAPE for Decision Tree= 18.812862571564235


# In[1008]:


#Linear Regression
model_lr = LinearRegression()#Initiate model
model_lr.fit(features_train , target_train)#fit on training data
pred_lr = model_lr.predict(features_test)#predict on test data
rmse_lr =np.sqrt(mean_squared_error(target_test,pred_lr))#Root mean square eror
mape_lr = np.mean((np.abs(target_test - pred_lr) / target_test))*100  #MAPE
print("MAPE for Linear Regression = "+str(mape_lr))
print("Root Mean Squared Error For Linear Regression = "+str(rmse_lr))
print("R^2 Score = "+str(r2_score(target_test,pred_lr)))


# In[1009]:


#Root Mean Squared Error For Linear Regression = 0.10414518826440598
#R^2 Score = 0.7726373756387745
#MAPE for Linear Regression = 18.268886233633456


# In[1010]:


#Random Forest
model_rf = RandomForestRegressor(random_state=1234)#Intiate model
model_rf.fit(features_train, target_train)#fit on training ata
pred_rf = model_rf.predict(features_test)#predict on test data


# In[1011]:


#Hyperparameter Tuning
params_rf = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [500, 1000]}
grid_rf = GridSearchCV(estimator=model_rf,
                       param_grid=params_rf,
                       n_jobs=-1, cv = 3)
grid_rf.fit(features_train, target_train)
print(grid_rf.best_estimator_)
model_rf_best = grid_rf.best_estimator_
pred_rf_best = model_rf_best.predict(features_test)
rmse_rf_best =np.sqrt(mean_squared_error(target_test,pred_rf_best))
mape_rf_best = np.mean((np.abs(target_test - pred_rf_best) / target_test))*100
print("Root Mean Squared Error For Random Forest= "+str(rmse_rf_best))
print("R^2 Score For Random Forest= "+str(r2_score(target_test,pred_rf_best)))
print("MAPE for Random Forest= "+str(mape_rf_best))


# In[ ]:


#MAPE for Random Forest= 14.396712183975607
#Root Mean Squared Error For Random Forest= 0.08564579859271436
#R^2 Score For Random Forest= 0.8462366680827034

