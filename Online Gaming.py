#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("GamingStudy_data[1].csv",encoding='unicode_escape')
df.head() 


# In[4]:


df = df[['S. No.', 'Game', 'Hours', 'whyplay', 'Gender', 'Age', 'Work', 'Degree', 'Birthplace', 'Residence', 'Playstyle', 'GAD_T', 'SWL_T', 'SPIN_T']]
df.head()
df.shape


# In[5]:


df.describe()


# In[6]:


df.info


# In[7]:


df.dtypes


# In[8]:


df['Game'] = df['Game'].str.strip()
df


# In[9]:


df['Game'] = df['Game'].str.lower()

df


# In[10]:


#strip the whyplay column of the whitespace and lowercase the values
df['whyplay'] = df['whyplay'].str.strip()
df


# In[11]:


df['whyplay'] = df['whyplay'].str.lower()
df


# In[12]:


#change to categorical data
df['Game'] = df['Game'].apply(lambda x: 1 if x == 'skyrim' else 2 if x == 'world of warcraft' else 3 if x == 
                                        'league of legends' else 4 if x == 'starcraft 2' else 5 if x == 'counter strike' 
                                        else 6 if x == 'destiny' else 7 if x == 'diablo 3' else 8 if x == 'heroes of the storm'
                                        else 9 if x == 'hearthstone' else 10 if x == 'guild wars 2' else 11)
df


# In[13]:


#make whyplay a integer
print(df['Game'].unique())


# In[14]:


sns.countplot(x='Game', data=df)
plt.legend
plt.show()


# In[15]:


#change to categorical data
#winning = 1
#improving = 2
#relaxing = 3
#havingfun = 4
df['whyplay'] = df['whyplay'].apply(lambda x: 1 if x == 'winning' else 2 if x == 'improving' 
                                              else 3 if x == 'relaxing' else 4 if x == 'having fun' else 5)
df


# In[16]:


print(df['whyplay'].unique())


# In[17]:


sns.countplot(x='whyplay', data=df)
#plt.legend
plt.show()


# In[18]:


#strip the Gender column of the whitespace and lowercase the values
df['Gender'] = df['Gender'].str.strip()
df


# In[19]:


df['Gender'] = df['Gender'].str.lower()
df


# In[20]:


#change to categorical data
#'Skyrim = 1' 'World of Warcraft = 2' 'League of Legends = 3' 'Starcraft 2 = 4''Counter Strike = 5' 'Destiny = 6' 'Diablo 3 = 7' 'Heroes of the Storm = 8' 'Hearthstone = 9''Guild Wars 2= 10 'Other = 11'
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'male' else 2 if x == 'female' else 3)
df


# In[21]:


print(df['Gender'].unique())


# In[22]:


sns.countplot(x='Gender', data=df)
plt.show()


# In[23]:


print(df['Work'].unique())


# In[24]:


#strip the work column of the whitespace and lowercase the values
df['Work'] = df['Work'].str.strip()
df


# In[25]:


df['Work'] = df['Work'].str.lower()
df


# In[26]:


#change to categorical data
#'Skyrim = 1' 'World of Warcraft = 2' 'League of Legends = 3' 'Starcraft 2 = 4''Counter Strike = 5' 'Destiny = 6' 'Diablo 3 = 7' 'Heroes of the Storm = 8' 'Hearthstone = 9''Guild Wars 2= 10 'Other = 11'
df['Work'] = df['Work'].apply(lambda x: 1 if x == 'unemployed / between jobs' else 2 if x == 'employed' 
                                        else 3 if x == 'student at college / university' else 4 if x == 'student at school' 
                                        else 5)
df


# In[27]:


sns.countplot(x='Work', data=df)
plt.show()


# In[28]:


print(df['Degree'].unique())


# In[29]:


df['Degree'] = df['Degree'].str.strip()
df


# In[30]:


df['Degree'] = df['Degree'].str.lower()
df


# In[31]:


df['Degree'] =df['Degree'].apply(lambda x: 1 if x == 'bachelor\xa0(or equivalent)'else 2 if x == 'high school diploma (or equivalent)' else 3 if x == 'ph.d., psy. d., md (or equivalent)' else 4 if x == 'master\xa0(or equivalent)' else 5)
df


# In[32]:


print(df['Degree'].unique())


# In[33]:


sns.countplot(x='Degree', data=df)
plt.show()


# In[34]:


print(df['Birthplace'].unique())


# In[35]:


birthplace = df['Birthplace'].unique()


# In[36]:


#create a new dataframe to store the unique values
birthplace_df = pd.DataFrame(birthplace, columns=['Birthplace'])


# In[37]:


#crete a new column to store the index
birthplace_df['values'] = birthplace_df.index


# In[38]:


birthplace_df['Birthplace'] = birthplace_df['Birthplace'].str.strip()


# In[39]:


birthplace_df['Birthplace'] = birthplace_df['Birthplace'].str.lower()


# In[40]:


#strip the whyplay column of the whitespace and lowercase the values
df['Birthplace'] = df['Birthplace'].str.strip()
df


# In[41]:


df['Birthplace'] = df['Birthplace'].str.lower()
df


# In[42]:


df['Birthplace'] = df['Birthplace'].map(birthplace_df.set_index('Birthplace')['values'])


# In[43]:


print(df['Birthplace'].unique())


# In[44]:


sns.countplot(x='Birthplace', data=df)
plt.show()


# In[45]:


print(df['Residence'].unique())


# In[46]:


Residence = df['Residence'].unique()


# In[47]:


Residence_df = pd.DataFrame(Residence, columns=['Residence'])
#crete a new column to store the index
Residence_df['values'] = Residence_df.index


# In[48]:


Residence_df['Residence'] = Residence_df['Residence'].str.strip()


# In[49]:


Residence_df['Residence'] = Residence_df['Residence'].str.lower()


# In[50]:


df['Residence'] = df['Residence'].str.strip()
df


# In[51]:


df['Residence'] = df['Residence'].str.lower()
df


# In[52]:


df['Residence'] = df['Residence'].map(Residence_df.set_index('Residence')['values'])
df


# In[53]:


print(df['Residence'].unique())


# In[54]:


sns.countplot(x='Residence', data=df)
plt.show()


# In[55]:


print(df['Playstyle'].unique())


# In[56]:


df['Playstyle'] = df['Playstyle'].str.strip()


# In[57]:


df['Playstyle'] = df['Playstyle'].str.lower()


# In[58]:


df['Playstyle'] = df['Playstyle'].apply(lambda x: 1 if x == 'singleplayer' else 2 
                                                  if x == 'multiplayer - offline (people in the same room)' else 3 
                                                  if x == 'multiplayer - online - with strangers' else 4 
                                                  if x == 'multiplayer - online - with online acquaintances or teammates' else 5
                                                  if x == 'multiplayer - online - with real life friends' else 6)


# In[59]:


print(df['Playstyle'].unique())


# In[60]:


sns.countplot(x='Playstyle', data=df)
plt.show()


# In[61]:


print(df.shape)


# In[62]:


# dropping null values 
df = df.dropna(subset=['S. No.'])
df = df.dropna(subset=['Game'])
df = df.dropna(subset=['Hours'])
df = df.dropna(subset=['whyplay'])
df = df.dropna(subset=['Gender'])
df = df.dropna(subset=['Age'])
df = df.dropna(subset=['Work'])
df = df.dropna(subset=['Degree'])
df = df.dropna(subset=['Birthplace'])
df = df.dropna(subset=['Residence'])
df = df.dropna(subset=['Playstyle'])
df = df.dropna(subset=['GAD_T'])
df = df.dropna(subset=['SWL_T'])
df = df.dropna(subset=['SPIN_T'])
print(df.shape)


# In[63]:


df.dtypes


# # Detecting and removing the outliers in numerical data( hours, GAD_T,SPIN_TSWL_T)

# In[64]:


sns.boxplot(x=df['Hours'])
plt.show()


# In[65]:


df = df[df['Hours'] > 0]
df = df[df['Hours'] < 50]


# In[66]:


sns.boxplot(x=df['Hours'])
plt.show()


# In[67]:


sns.boxplot(x=df['Age'])
plt.show()


# In[68]:


sns.countplot(x='Age', data=df)
plt.show()


# In[69]:


sns.boxplot(x=df['GAD_T'])
plt.show()


# In[70]:


df = df[df['GAD_T'] > 0]
df = df[df['GAD_T'] < 15]


# In[71]:


sns.boxplot(x=df['GAD_T'])
plt.show()


# In[72]:


sns.boxplot(x=df['SPIN_T'])
plt.show()


# In[73]:


df = df[df['SPIN_T'] > 0]
df = df[df['SPIN_T'] < 60]


# In[74]:


sns.boxplot(x=df['SPIN_T'])
plt.show()


# In[75]:


sns.boxplot(x=df['SWL_T'])
plt.show()


# #  Normalisation of numerical data

# In[76]:


temp_data = np.array(df['Hours'])


# In[77]:


mean = np.mean(temp_data, axis = 0)
print("")
print("The mean of each variable:",mean)
sd = np.std(temp_data, axis = 0)
print("")
print("The Standard Deviation of each variable:",sd)


# In[78]:


final_data = [x for x in temp_data if (x > mean - 2 * sd)]
final_data = [x for x in final_data if (x < mean + 2 * sd)]

plt.hist(final_data)
plt.show()


# In[79]:


df['Hours']


# In[80]:


temp_data = np.array(df['GAD_T'])


# In[81]:


mean = np.mean(temp_data, axis = 0)
print("")
print("The mean of each variable:",mean)
sd = np.std(temp_data, axis = 0)
print("")
print("The Standard Deviation of each variable:",sd)


# In[82]:


final_data = [x for x in temp_data if (x > mean - 2 * sd)]
final_data = [x for x in final_data if (x < mean + 2 * sd)]

plt.hist(final_data)
plt.show()


# In[83]:


temp_data = np.array(df['Age'])


# In[84]:


mean = np.mean(temp_data, axis = 0)
print("")
print("The mean of each variable:",mean)
sd = np.std(temp_data, axis = 0)
print("")
print("The Standard Deviation of each variable:",sd)


# In[85]:


final_data = [x for x in temp_data if (x > mean - 2 * sd)]
final_data = [x for x in final_data if (x < mean + 2 * sd)]

plt.hist(final_data)
plt.show()


# In[86]:


temp_data = np.array(df['SWL_T'])


# In[87]:


mean = np.mean(temp_data, axis = 0)
print("")
print("The mean of each variable:",mean)
sd = np.std(temp_data, axis = 0)
print("")
print("The Standard Deviation of each variable:",sd)


# In[88]:


final_data = [x for x in temp_data if (x > mean - 2 * sd)]
final_data = [x for x in final_data if (x < mean + 2 * sd)]

plt.hist(final_data)
plt.show()


# In[89]:


from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# In[90]:


# X_value= df[['Hours', 'Age']]


# In[91]:


df_temp=df.copy()


# In[92]:


df_temp['Hours']=df_temp['Hours']/df_temp['Hours'].abs().max()


# In[93]:


sns.distplot(df_temp['Hours'])


# In[94]:


# data = pd.DataFrame(X_value)
df['Hours']=df_temp['Hours']


# In[95]:


df


# In[96]:


df_temp=df.copy()


# In[97]:


df_temp['Age']=df_temp['Age']/df_temp['Age'].abs().max()


# In[98]:


sns.distplot(df_temp['Age'])


# In[99]:


df['Age']=df_temp['Age']


# In[100]:


df


# In[101]:


print(df.corr())
# print("")


# In[102]:


sns.heatmap(df.corr(), annot=True)
plt.savefig("heatmap.png")
plt.figure(figsize=(10,200))
plt.show()


# In[103]:


df.corr()


# # Rejection of null hypothsis 

# In[104]:


from scipy.stats import ttest_ind
#test the null hypothesis that the data is from a normal distribution
print("T-test")
t_test = ttest_ind(df['SPIN_T'],df['GAD_T'])
print(t_test)

##### Since the p<0.05 and we can reject the  null hypothesis and it is not normally distributed.


# # Training models for Random Forest 

# In[105]:


x = df[['Game', 'Hours', 'whyplay', 'Gender', 'Age', 'Work', 'Degree', 'Birthplace', 'Residence', 'Playstyle', 'SWL_T', 'SPIN_T']]
y = df['GAD_T']


# In[106]:


x = x.apply(pd.to_numeric, errors='coerce')


# In[107]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[108]:


# spliting the data for training set 70 and testing set 30
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[109]:


from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(x_train, y_train)
RF_pred = RF_model.predict(x_test)


# In[110]:


print("Random Forest Model")
print('Mean squared error: %.2f' % mean_squared_error(y_test, RF_pred))  
print('Coefficient of determination: %.2f' % r2_score(y_test, RF_pred))


# In[111]:


from sklearn.metrics import accuracy_score
RF_accuracy = accuracy_score(y_test, RF_pred) * 100
print('Accuracy: ', RF_accuracy)


# In[155]:


#create a bar plot of the model performance
from sklearn.metrics import classification_report
print(classification_report(y_test, RF_pred))
# print("")


# In[ ]:


imp = mdl.feature_importances_
    importance_dct = dict(zip(ft_cols, imp))
    
    if viz:
        plt.clf()
        plt.barh(list(importance_dct.keys()), list(importance_dct.values()))
        plt.xlabel("Tree Based Importance")
        plt.title("Tree Based Feature Importance")
        plt.show()
    
    return importance_dct
    
ti = tree_feature_importance(gbr, ft_cols, True)


# In[113]:


#create a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, RF_pred)
sns.heatmap(cm, annot=True)
# plt.figure(figsize=(10,16))
plt.savefig("heatmap for model 1.png")
plt.show()


# In[114]:


x2 = df[['Game', 'Hours', 'whyplay', 'Gender', 'Age', 'Work', 'Degree', 'Birthplace', 'Residence', 'Playstyle', 'SWL_T', 'GAD_T']]
y2 = df['SPIN_T']


# In[115]:


x2 = x2.apply(pd.to_numeric, errors='coerce')


# In[116]:


# spliting the data for training set 70 and testing set 30
x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size = 0.3, random_state = 0)


# In[117]:


print("Random Forest Model")
print('Mean squared error: %.2f' % mean_squared_error(y_test, RF_pred))  
print('Coefficient of determination: %.2f' % r2_score(y_test, RF_pred))


# In[118]:


RF_accuracy = accuracy_score(y_test, RF_pred) * 100
print('Accuracy: ', RF_accuracy)


# In[119]:


#create a bar plot of the model performance
from sklearn.metrics import classification_report
print(classification_report(y_test, RF_pred))
print("")


# In[120]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, RF_pred)
sns.heatmap(cm, annot=True)
plt.savefig("heatmap for model 2.png")
plt.show()


# # Gradient Boosting for GAD_T

# In[159]:


from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance
# import numpy as np
# import pandas as pd


# In[160]:


xg = df[['Game', 'Hours', 'whyplay', 'Gender', 'Age', 'Work', 'Degree', 'Birthplace', 'Residence', 'Playstyle', 'SPIN_T']]
yg = df['GAD_T']


# In[161]:


# spliting data into 30/70
train_X, test_X, train_y, test_y = train_test_split(xg, yg, test_size=0.30,random_state=1)


# In[162]:


import numpy as np
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


# In[125]:


from sklearn.ensemble import GradientBoostingRegressor


# In[163]:


GR = GradientBoostingRegressor(n_estimators = 200, max_depth = 1, random_state = 1) 
gmodel = GR.fit(train_X, train_y) 
g_predict = gmodel.predict(test_X)
GB_MAPE = MAPE(test_y,g_predict)
Accuracy = 100 - GB_MAPE
print("MAPE: ",GB_MAPE)
print('Accuracy of Linear Regression: {:0.2f}%.'.format(Accuracy))


# In[166]:


def permutation_feature_importance(model, X, y, xg, viz = True):
    pi = permutation_importance(gmodel, X, y)
    importance_dct = dict(zip(xg, pi.importances_mean))
    
    if viz:
        plt.clf()
        plt.barh(list(importance_dct.keys()), list(importance_dct.values()))
        plt.xlabel("Permutation Importance")
        plt.title("Permutation Feature Importance")
        plt.show()
    return importance_dct
    
imp_dct = permutation_feature_importance(gmodel,xg,yg,xg, viz = True)


# # Gradient Boosting for SPIN_T 

# In[171]:


xg = df[['Game', 'Hours', 'whyplay', 'Gender', 'Age', 'Work', 'Degree', 'Birthplace', 'Residence', 'Playstyle', 'GAD_T']]
yg = df['SPIN_T']


# In[172]:


train_X, test_X, train_y, test_y = train_test_split(xg, yg, test_size=0.30,random_state=1)


# In[173]:


import numpy as np
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


# In[174]:


from sklearn.ensemble import GradientBoostingRegressor


# In[175]:


GR = GradientBoostingRegressor(n_estimators = 200, max_depth = 1, random_state = 1) 
gmodel = GR.fit(train_X, train_y) 
g_predict = gmodel.predict(test_X)
GB_MAPE = MAPE(test_y,g_predict)
Accuracy = 100 - GB_MAPE
print("MAPE: ",GB_MAPE)
print('Accuracy of Linear Regression: {:0.2f}%.'.format(Accuracy))


# In[176]:


def permutation_feature_importance(model, X, y, xg, viz = True):
    pi = permutation_importance(gmodel, X, y)
    importance_dct = dict(zip(xg, pi.importances_mean))
    
    if viz:
        plt.clf()
        plt.barh(list(importance_dct.keys()), list(importance_dct.values()))
        plt.xlabel("Permutation Importance")
        plt.title("Permutation Feature Importance")
        plt.show()
    return importance_dct
    
imp_dct = permutation_feature_importance(gmodel,xg,yg,xg, viz = True)


# # Logistic Regression  for GAD_T

# In[132]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


# In[133]:


xl = df[['Game', 'Hours', 'whyplay', 'Gender', 'Age', 'Work', 'Degree', 'Birthplace', 'Residence', 'Playstyle', 'SWL_T', 'SPIN_T']]
yl = df['GAD_T']


# In[134]:


train_X, test_X, train_y, test_y = train_test_split(xl, yl, test_size=0.30,random_state=1)


# In[135]:


#instantiate the model
log_regression = LogisticRegression()


# In[136]:


#fit the model using the training data
log_regression.fit(train_X,train_y)


# In[137]:


#use model to make predictions on test data
y_pred = log_regression.predict(test_X)


# In[138]:


cnf_matrix = metrics.confusion_matrix(test_y, y_pred)
cnf_matrix


# In[139]:


print("Accuracy:",metrics.accuracy_score(test_y, y_pred))


# In[140]:


from scikitplot.metrics import roc_curve


# In[141]:


# from scikitplot.metrics import plot_roc_curve
y_pred_proba = log_regression.predict_proba(test_X)[::,1]
fpr, tpr, thresholds = roc_curve(test_y,y_pred,pos_label=1)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend
plt.savefig("ROC.png")
plt.show()


# # Logistic Regression for SPIN_T

# In[142]:


xl = df[['Game', 'Hours', 'whyplay', 'Gender', 'Age', 'Work', 'Degree', 'Birthplace', 'Residence', 'Playstyle', 'SWL_T','GAD_T']]
yl = df['SPIN_T']


# In[143]:


train_X, test_X, train_y, test_y = train_test_split(xl, yl, test_size=0.30,random_state=1)


# In[144]:


#instantiate the model
log_regression = LogisticRegression()


# In[145]:


#fit the model using the training data
log_regression.fit(train_X,train_y)


# In[146]:


#use model to make predictions on test data
y_pred = log_regression.predict(test_X)


# In[147]:


cnf_matrix = metrics.confusion_matrix(test_y, y_pred)
cnf_matrix


# In[148]:


print("Accuracy:",metrics.accuracy_score(test_y, y_pred))


# In[149]:


from scikitplot.metrics import roc_curve


# In[150]:


y_pred_proba = log_regression.predict_proba(test_X)[::,1]
fpr, tpr, thresholds = roc_curve(test_y,y_pred,pos_label=1)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend
plt.savefig("ROC 2.png")
plt.show()


# # Hypothesis testing

# In[151]:


from scipy import stats


# In[152]:


stats.normaltest(df['GAD_T'])


# In[153]:


stats.normaltest(df['SPIN_T'])


# In[154]:


# Coreelation between hours and GAD_T
stat_pearson, p_pearson = stats.pearsonr(df['Hours'],df['GAD_T'])
print(' Pearson - coefficent: ', stat_pearson, 'P-value: ',p_pearson)


# In[ ]:





# In[ ]:





# In[ ]:




