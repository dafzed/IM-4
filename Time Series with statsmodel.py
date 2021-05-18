#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import sys, os
import warnings
warnings.simplefilter(action='ignore')


# In[2]:


import fbprophet
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[3]:


import seaborn as sns
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree


# In[4]:


df = pd.read_csv('DataML2.csv')


# In[5]:


df['period'] = pd.to_datetime(df['period'], format='%b-%y')
df


# In[6]:


df.dtypes


# In[7]:


df.index = df['period']
df.index.name = None
df = df.drop(columns=['period'])
df


# In[8]:


df.index.name='period'
df


# In[10]:


data = df.drop(columns=['csplrl','csprl','cslrl','csgrl','invrl','xgsrl','mgsrl','inhrl','inbrl','retailsales','sukucadang','manminrok','bbm','gadget','perlengkapanrt','baranglain','onlinesales','onlinetranspr','motorsales','konsumsiskn','tariktunai','belanjadebet','tariktunaicc','ntp','wageburuh','wageart','wagebarber','wagetukang','ihsg','yield1y','IHPRsekunder','DPKindividu','KreditKonsumsi','JobVacancy','IEK','jalantolgol1','inflasipendidikan','inflasikesehatan','devisa_transpor','devisa_travel','semensales','imporkonstruksi','orderbook','voltraffgol3','voltraffgol5','presalesprop','pmi','salesalatberat','kredit_mk','kredit_inv','voltraffgol4','rtgs','ekspor_nm_rl','impor_nm_rl','ekspor_nm_pi','impor_nm_pi','ekspor_mg_nl','impor_mg_nl','sup_valas_jl','sup_valas_bl','goog_retail','goog_groc','goog_parks','goog_transit','goog_work','goog_res','goog_avg','IPBK','IPAMM','ITP','IIK'])


# In[12]:


data = data.pct_change(periods=12)[['barangbudaya','barangsandang','mobilsales','IKK','IKE','prod_motor']]
data


# In[13]:


gdprl = df.loc['2013-01-01':, 'gdprl']


# In[14]:


data['gdprl'] = gdprl
data = data.dropna()
data


# In[15]:


data2 = data.drop(columns=['barangbudaya','barangsandang','mobilsales','IKK','IKE','prod_motor'])
data2.head()


# In[27]:


data.index.freq='MS'
data2.index.freq='MS'
data2.head()


# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['gdprl'])
result.plot()
plt.show()


# In[19]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[20]:


adf_test(data['gdprl'])


# In[22]:


from statsmodels.graphics.tsaplots import month_plot, quarter_plot
month_plot(data['gdprl']);


# In[24]:


dfq = data['gdprl'].resample(rule='Q').mean()
quarter_plot(dfq);


# In[30]:


len(data)


# In[32]:


# Set one year for testing
train = data.iloc[:84]
test = data.iloc[84:]


# In[42]:


traindata = data2.iloc[:84]
testdata = data2.iloc[84:]


# In[43]:


traindata


# In[34]:


test


# In[35]:


train


# In[37]:


data['gdprl'].plot(figsize=(12,8));


# # ExponentialSmoothing

# In[38]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[44]:


fitted_model = ExponentialSmoothing(traindata, trend='add', seasonal='add',
                                   seasonal_periods= 12).fit()


# In[46]:


test_predictions = fitted_model.forecast(12)
test_predictions


# In[51]:


#traindata.plot(legend=True, label='Train', figsize=(12,8))
testdata.plot(legend=True, label='Test', figsize=(12,8))
test_predictions.plot(legend=True, label='Prediction')


# In[49]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
np.sqrt(mean_squared_error(testdata, test_predictions))


# In[50]:


testdata.describe()


# In[52]:


# FORECASTING INTO FUTURE
final_model = ExponentialSmoothing(traindata, trend='add', seasonal='add',
                                   seasonal_periods= 12).fit()


# In[53]:


forecast_predictions = final_model.forecast(36)


# In[54]:


data2.plot(figsize=(12,8))
forecast_predictions.plot();


# # ARIMA

# In[57]:


from pmdarima import auto_arima
auto_arima(data['gdprl'],seasonal=True).summary()


# # Before First Difference

# In[60]:


from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
model = ARMA(traindata,order=(2,2))
results = model.fit()
results.summary()


# In[61]:


start=len(traindata)
end=len(traindata)+len(testdata)-1
predictions = results.predict(start=start, end=end).rename('ARMA(2,2) Predictions')


# In[62]:


title = 'GDP'
ylabel='Gdprl'
xlabel='' # we don't really need a label here

ax = testdata.plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# In[65]:


stepwisefit = auto_arima(data['gdprl'], start_p=0, start_q=0,
                          max_p=2, max_q=2, m=12,
                          seasonal=False,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwisefit.summary()


# In[66]:


model = ARIMA(traindata,order=(0,1,0))
results = model.fit()
results.summary()


# In[68]:


predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(0,1,0) Predictions')


# In[69]:


predictions


# In[73]:


# Plot predictions against known values
title = 'GDP'
ylabel='gdprl'
xlabel='' # we don't really need a label here

ax = testdata.plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
#ax.yaxis.set_major_formatter(formatter);


# In[75]:


from statsmodels.tsa.statespace.tools import diff
datafd = diff(data['gdprl'],k_diff=1)

# Equivalent to:
# df1['d1'] = df1['Inventories'] - df1['Inventories'].shift(1)

adf_test(datafd,'Real Manufacturing and Trade Inventories')


# In[78]:


datafd


# In[77]:


datafd.plot()
data['gdprl'].plot()


# In[84]:


stepwise_fit = auto_arima(data['gdprl'], start_p=0, start_q=0,
                          max_p=6, max_q=6, m=12,
                          seasonal=True,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# In[86]:


stepwise_fit = auto_arima(data2, m=12,
                          seasonal=True,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# In[87]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[94]:


model = SARIMAX(traindata, order=(1,1,1), seasonal_order=(0,1,0,12),
               enforce_invertibility=False)


# In[95]:


results = model.fit()


# In[96]:


results.summary()


# In[99]:


predictions = results.predict(start,end).rename('SARIMA Model')
predictions


# In[100]:


testdata


# In[98]:


testdata.plot(legend=True, figsize=(15,8))
predictions.plot(legend=True)


# In[101]:


from fbprophet import Prophet


# In[107]:


data2 = data2.reset_index() #.set_index('period')
data2.drop(columns=['index'])


# In[110]:


data2 = data2.drop(columns=['index','level_0'])
data2


# In[112]:


data2.columns = ['ds','y']
data2.head()


# In[134]:


m = Prophet()
m.fit(data2)


# In[133]:


get_ipython().system(' pip install PYSTAN')


# In[132]:


get_ipython().system(' pip install CMDSTANPY')


# In[117]:


get_ipython().system(' pip install PyInstaller')


# In[118]:


from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('fbprophet')
datas = collect_data_files('fbprophet')


# In[136]:


from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('PYSTAN')
datas = collect_data_files('PYSTAN')


# In[121]:


from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('Cython')
datas = collect_data_files('Cython')


# In[124]:


datas


# In[129]:


hiddenimports = ['stan_backend']
hiddenimports = Prophet()


# In[137]:


from fbprophet import Prophet
import logging
logger = logging.getLogger('fbprophet')
logger.setLevel(logging.DEBUG)

m = Prophet()
print(m.stan_backend)


# In[3]:


import sys
sys.version_info

