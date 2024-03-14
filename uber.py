#!/usr/bin/env python
# coding: utf-8

# ### In this notebook i will do some analysis and simple regression model based on our data which is uber and lyft fare dataset. I got this data from kaggle where the main purpose and objective of this large dataset is to model how price or cab fare varies with all the features that've been given.

#  

# ### If the output code can't be seen in github, this whole notebook can also be accessed in my kaggle notebook:
# ### https://www.kaggle.com/danielbeltsazar/uber-lyft-price-prediction

#  

# # 1. Importing Library and Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
pd.options.display.max_rows = None
pd.options.display.max_columns = None


# In[2]:


df=pd.read_csv("rideshare_kaggle.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df['datetime']=pd.to_datetime(df['datetime'])


# In[5]:


df.isnull().sum().sum()


# In[6]:


df.dropna(axis = 0,inplace = True)


# In[7]:


df.isnull().sum().sum()


# In[8]:


df['visibility'].head()


# In[9]:


df['visibility.1'].head()


# In[10]:


df = df.drop(['visibility.1'],axis=1)


# # 2. EDA and Visualization

# ## 1. Time Analysis 

# ### --Month Data--

# In[11]:


import matplotlib.pyplot as plt
import plotly.express as px


# In[12]:




# In[13]:


fig = px.line(x=[1,2, 3], y=[1, 2, 3]) 
 
# printing the figure instance
print(fig)


# In[ ]:





# In[14]:


#pip install plotly


# In[15]:


#pip install plotly --upgrade


# In[16]:


#pip install -U kaleido


# In[17]:


import matplotlib.pyplot as plt

def plot_bar(groupby_column):
    df1 = df.groupby(groupby_column).size().reset_index(name="counts")
    
    plt.figure(figsize=(10, 6))
    plt.bar(df1[groupby_column], df1["counts"], color='green')
    plt.xlabel(groupby_column)
    plt.ylabel('Counts')
    plt.title('Bar plot by ' + groupby_column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[18]:




def plot_bar(groupby_column):
    df1 =df.groupby(groupby_column).size().reset_index(name="counts")
    fig1 = px.bar(data_frame=df1, x=groupby_column, y="counts", color=groupby_column, barmode="group")
    print(df1)
    fig1.show(renderer='png')


# In[ ]:





# In[19]:


plot_bar('month')


# ### It appears that we only have november and december in our month data. It means the data is only recorded or taken in november and december with december data dominating.

# ### --Day Data--

# In[20]:


plot_bar('day')


# ### It seems we have many gaps in our 'day' data. For example we don't have data from 18th day until 25th day in each month.

# ### --Hour Data--

# In[21]:


plot_bar('hour')


# ### It seems we have almost 24 hours recorded data

# ## 2. Source and Destination Analysis

# In[22]:


plot_bar('source')


# ### It seems that all sources are almost equal in number. There are about 50k data in each source feature (Back Bay, Beacon Hill, Boston University, etc)

# In[23]:


plot_bar('destination')


# ### Same with source feature, there are about 50k data in each destination feature (Back Bay, Beacon Hill, Boston University, etc)

# In[24]:


df.groupby(by=["destination","source"]).agg({'latitude':'mean','longitude':'mean'})


# ### Here i attached the example image of the map plot of one of the cab trip routes. I can't render it here because it won't be available to see in github. For a complete visualization you can see my notebook that includes all the maps in my Geospatial Project repository

# In[25]:


#pip install geopandas


# In[26]:


#pip install folium


# In[27]:


from folium.plugins import FastMarkerCluster


# In[28]:


import geopandas as gpd
import folium
df1 = df[df['source']=='Haymarket Square']
my_map = folium.Map(location=[df1["latitude"].mean(), df1["longitude"].mean()],zoom_start = 10)
my_map.add_child(FastMarkerCluster(df1[['latitude', 'longitude']].values.tolist(),color='green'))
my_map


# ![Taxi1.png](attachment:Taxi1.png)

# ### We can see that trips which their sources are Haymarket Square have two groups or clusters of destination that contain many places (we can see them if we zoom the map). Many of them are in boston area as we can see that there are 46256 data in that cluster.

# ## 3. Cab Type Analysis

# In[29]:


plot_bar('cab_type')


# ### So for our whole data, we have uber data more than lyft data. The difference is not too big, each cab type has about 300K data.

# In[30]:


import matplotlib.pyplot as plt

def plot_grouped_bar(df, x_column, y_column, group_column):
    # Grouping the DataFrame by 'x_column' and 'group_column' and aggregating the counts
    grouped_df = df.groupby([x_column, group_column]).size().unstack(fill_value=0).reset_index()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35  # Width of the bars
    x = range(len(grouped_df[x_column]))  # X-axis values

    # Plotting each group's bars
    for i, col in enumerate(grouped_df.columns[1:]):
        ax.bar([p + width * i for p in x], grouped_df[col], width, label=col)

    ax.set_xlabel(x_column)
    ax.set_ylabel('Counts')
    ax.set_title('Grouped Bar Plot')
    ax.set_xticks([p + 0.5 * width for p in x])
    ax.set_xticklabels(grouped_df[x_column], rotation=45)
    ax.legend(title=group_column)
    plt.tight_layout()
    plt.show()

# Call the function with the appropriate arguments
plot_grouped_bar(df, 'day', 'counts', 'cab_type')



#f2 =df.groupby(by=["day","cab_type"]).size().reset_index(name="counts")
#fi2 = px.ar(data_frame=df2, x="day", y="counts", color="cab_type", barmode="group")
#fig2.show(renderer='png')


# In[31]:


import matplotlib.pyplot as plt

def plot_grouped_bar(df, x_column, y_column, group_column):
    # Grouping the DataFrame by 'x_column' and 'group_column' and aggregating the counts
    grouped_df = df.groupby([x_column, group_column]).size().unstack(fill_value=0).reset_index()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35  # Width of the bars
    x = range(len(grouped_df[x_column]))  # X-axis values

    # Plotting each group's bars
    for i, col in enumerate(grouped_df.columns[1:]):
        ax.bar([p + width * i for p in x], grouped_df[col], width, label=col)

    ax.set_xlabel(x_column)
    ax.set_ylabel('Counts')
    ax.set_title('Grouped Bar Plot')
    ax.set_xticks([p + 0.5 * width for p in x])
    ax.set_xticklabels(grouped_df[x_column], rotation=45)
    ax.legend(title=group_column)
    plt.tight_layout()
    plt.show()

# Call the function with the appropriate arguments
plot_grouped_bar(df, 'hour', 'counts', 'cab_type')





#df3 =df.groupby(["hour","cab_type"]).size().reset_index(name="counts")
#fig3 = px.bar(data_frame=df3, x="hour", y="counts", color="cab_type", barmode="group")
#fig3.show(renderer='png')


# ### So in every day and every hour recorded, uber seems dominating booking order in our data

# In[ ]:





# ## 4. Price Analysis

# ### We can see average or mean of our price data in every route (source-destination) through table below

# In[32]:


pd.set_option('display.max_rows', 72)
df.groupby(by=["source","destination"]).price.agg(["mean"])


# ### And we can see our maximum price data

# In[33]:


print('Maximum price in our data :',df.price.max())
df[df['price']==df.price.max()]


# In[34]:


df[df['price']==df.price.max()][['latitude','longitude']]


# ### I can plot the map of both places using folium to see how far they are from each other (I only inserted the snapshot of the plot)

# In[35]:


# Using this code:
map1 = folium.Map(location=(42.3503,-71.081),zoom_start = 10)
folium.Marker(location=(42.3503,-71.081)).add_to(map1) # Fenway
folium.Marker(location=(42.3378,-71.066)).add_to(map1) # Financial District
display(map1)


# ### Apparently the 'Financial District - Fenway' route (by lyft) costs 97.5 dollars, which is our maximum price data. But from the map above, the distance between both places is not too far (they are both in boston), so it could be outlier since we don't have information about trip duration or transit. We should check another data with the same route

# In[36]:


df_group = df.groupby(by=["source","destination"]).price.agg(["mean"]).reset_index()
df_group[(df_group['source']=='Financial District')& (df_group['destination']=='Fenway')]


# ### The mean of the price data of that route is 23.4 dollars, which is far from our maximum price data (97.5 dollars). Then it is possible an outlier. We can drop it.

# In[37]:


df = df.loc[df['price']!=df.price.max()]


# In[38]:


df.head()


# In[ ]:





# # 3. Data Preprocessing / Feature Engineering

# ## 1. Removing Unnecessary Features

# In[39]:


# For further modelling i don't think we need date related features. But maybe we need them in the future analysis.
# so i will make new dataframe

new_df = df.drop(['id','timestamp','datetime','long_summary','apparentTemperatureHighTime','apparentTemperatureLowTime',
                  'apparentTemperatureLowTime','windGustTime','sunriseTime','sunsetTime','uvIndexTime','temperatureMinTime',
                 'temperatureMaxTime','apparentTemperatureMinTime','temperatureLowTime','apparentTemperatureMaxTime'],axis=1)


# In[40]:


new_df.shape


# ### Our goal is to make linear regression model. First we check correlation between our features and target feature (price)

# ### First, i want to check the correlation of our temperature related features with our target feature (Price)

# In[41]:


temp_cols= ['temperature','apparentTemperature','temperatureHigh','temperatureLow','apparentTemperatureHigh',
                'apparentTemperatureLow','temperatureMin','temperatureHighTime','temperatureMax','apparentTemperatureMin','apparentTemperatureMax','price']


# In[42]:


df_temp = new_df[temp_cols]
df_temp.head()


# In[43]:


plt.figure(figsize=(15,20))
sns.heatmap(df_temp.corr(),annot=True)


# ### We see that all temperature related features have weak correlation with our target feature which is price
# 
# ### Removing all of them will not make any impact to our regression model

# In[44]:


new_df = new_df.drop(['temperature','apparentTemperature','temperatureHigh','temperatureLow','apparentTemperatureHigh',
                'apparentTemperatureLow','temperatureMin','temperatureHighTime','temperatureMax','apparentTemperatureMin','apparentTemperatureMax'],axis=1)
new_df.shape


#  

# ### Second, i want to check the correlation of our cilmate related features with our target feature (Price)

# In[45]:


climate_column = ['precipIntensity', 'precipProbability', 'humidity', 'windSpeed',
       'windGust', 'visibility', 'dewPoint', 'pressure', 'windBearing',
       'cloudCover', 'uvIndex', 'ozone', 'moonPhase',
       'precipIntensityMax','price']
df_clim = new_df[climate_column]
df_clim.head()


# In[46]:


plt.figure(figsize=(15,20))
sns.heatmap(df_clim.corr(),annot=True)


# ### Apparently all climate related features also have weak correlation with our target feature which is price
# 
# ### Once again, removing all of them will not make any impact to our regression model

# In[47]:


new_df = new_df.drop(['precipIntensity', 'precipProbability', 'humidity', 'windSpeed',
       'windGust', 'visibility', 'dewPoint', 'pressure', 'windBearing',
       'cloudCover', 'uvIndex', 'ozone', 'moonPhase',
       'precipIntensityMax'],axis=1)
new_df.shape


#  

# ### Third, i want to check our categorical value in our dataset features 

# In[48]:


category_col = new_df.select_dtypes(include=['object','category']).columns.tolist()
for column in new_df[category_col]:
    print(f'{column} : {new_df[column].unique()}')
    print()


# ### We can see that 'timezone' feature has only 1 value and 'product_id' feature contains many unidentified values. So we can remove or drop them.

# In[49]:


new_df = new_df.drop(['timezone','product_id'],axis=1)


# In[50]:


new_df.shape


#  

# ### Fourth, i want to check the correlation of our categorical features with our target feature (price)

# In[51]:


new_cat = ['source',
 'destination',
 'cab_type',
 'name',
 'short_summary',
 'icon','price']

df_cat = new_df[new_cat]
df_cat.head()


# In[52]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df_cat_encode= df_cat.copy()
for col in df_cat_encode.select_dtypes(include='O').columns:
    df_cat_encode[col]=le.fit_transform(df_cat_encode[col])


# In[53]:


df_cat_encode


# In[54]:


plt.figure(figsize=(15,20))
sns.heatmap(df_cat_encode.corr(),annot=True)


# ### We can see only name feature that has a relatively strong correlation. Source,destination, and cab_type features have relatively weak correlation, but i will pick cab_type feature because it has stronger correlation than other two features. I will drop or remove the rest of the columns

# In[55]:


new_df = new_df.drop(['source','destination','short_summary','icon'],axis=1)
new_df.head()


# ### Also i will remove hour, day, month, latitude, longitude, because we won't need them for now

# In[56]:


new_df = new_df.drop(['hour','day','month','latitude','longitude'],axis=1)
new_df.head()


# In[57]:


new_df.columns


# ## 2. Removing Outliers

# ### We've already done this before but only to one instance which has maximum price value. We want to check another possible outlier.

# ### We're using IQR method for checking top and bottom outliers

# In[58]:


Qp12 = new_df['price'].quantile(0.25)
Qp32 = new_df['price'].quantile(0.75)
IQRp = Qp32-Qp12


# In[59]:


new_df[new_df['price']>(Qp32+(1.5*IQRp))]


# In[60]:


new_df[new_df['price']<(Qp12-(1.5*IQRp))]


# ### We can see that we have 5588 data outliers. We can remove or drop them.

# In[61]:


print('Size before removing :',new_df.shape)
new_df= new_df[~((new_df['price']>(Qp32+(1.5*IQRp))))]
print('Size after removing :',new_df.shape)


# In[ ]:





# # 4. Regression Model

# ## 1. Encoding Data (One Hot Encoding)

# In[62]:


def one_hot_encoder(data,feature,keep_first=True):

    one_hot_cols = pd.get_dummies(data[feature])
    
    for col in one_hot_cols.columns:
        one_hot_cols.rename({col:f'{feature}_'+col},axis=1,inplace=True)
    
    new_data = pd.concat([data,one_hot_cols],axis=1)
    new_data.drop(feature,axis=1,inplace=True)
    
    if keep_first == False:
        new_data=new_data.iloc[:,1:]
    
    return new_data


# In[63]:


new_df_onehot=new_df.copy()
for col in new_df_onehot.select_dtypes(include='O').columns:
    new_df_onehot=one_hot_encoder(new_df_onehot,col)
    
new_df_onehot.head()


#  

# ## 2. Dataset Split

# In[64]:


from sklearn.model_selection import train_test_split
X = new_df_onehot.drop(columns=['price'],axis=1).values
y = new_df_onehot['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#  

# ## 3. Modeling

# ## 3.1. Base Model

# In[65]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()


# In[66]:


# Fit to data training
model = reg.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[67]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[68]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(mse)
print(rmse)


# ### Then for the long journey we have done, we got our regression model with accuracy or score 93.37% and RMSE value 2.26. It's not the best score though, we still can improve it with other regression models which could give better results.

# ## 3.2. Finding Best Models with best configuration with GridSearch CV

# In[69]:


from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV,ShuffleSplit

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

import warnings
warnings.filterwarnings('ignore')

find_best_model_using_gridsearchcv(X,y)


# ### Here we got our best model is decision tree regressor with r-squared 0.964, higher than our linear regression before.

# In[ ]:





# In[70]:


# Selecting the subset of features
selected_features = ['distance', 'surge_multiplier', 'temperature', 'humidity', 'price']

# Create the feature matrix X and target vector y
X = df[selected_features].values
y = df['price'].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import mean_squared_error
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Save the model
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(reg, file)


# In[ ]:





# In[ ]:




