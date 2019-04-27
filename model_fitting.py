# Utils
import pandas as pd
from itertools import cycle
from datetime import datetime
import numpy as np

# Modeling
from sklearn.linear_model import ElasticNet, enet_path, lasso_path, BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_model import ARMA
from sklearn import gaussian_process as gp
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.utils import resample

# Loading data
gdp = pd.read_csv('gdp_growth.csv')

X = pd.read_csv("data.csv").drop('date', axis = 1)

# Defining time periods
start = datetime(2009,4,1)
first = np.where(gdp.date == start.strftime('%Y-%m-%d'))[0][0]
last = gdp.shape[0]
length = last - first

# Preallocating results data frame
results = gdp.iloc[first:, :]
results.rename(columns = {'date' : 'Date', 'gdp' : 'Actual'}, inplace = True)
models = ['AR', 'GP', 'BLR', 'EN', 'Ridge']
for model in models:
    results[model] = np.zeros(length)
    results[model+'_std'] = np.zeros(length)

# Fitting models and forecasting one quarter out
while (first < last):
    
    # update training and test data sets
    y_train = gdp.gdp[:first]
    X_train = X.iloc[:first, :]
    X_test = X.iloc[first, :].values.reshape(1,-1)

    ## ARMA ##################################################################
    
    # Initializing AR(2,0) model
    ar = ARMA(y_train, order = (2,0))
    
    # Fitting model
    ar_model = ar.fit(transparams = False, disp = False)
    
    # Predict and store next quarter GDP growth
    results.loc[first, 'AR'] = ar_model.forecast()[0][0]
    results.loc[first, 'AR_std'] = ar_model.forecast()[1][0]
    
    ## GP ####################################################################
    
    # Gaussian Kernel with constant term and white noise
    kernel = ConstantKernel() + ConstantKernel() * RBF() + WhiteKernel()
    
    # Initializing GP model (normalized y)
    gp_model = gp.GaussianProcessRegressor(kernel = kernel, 
                                           n_restarts_optimizer = 20,
                                           normalize_y = True)
    
    # Fitting model
    gp_model.fit(X_train, y_train)
    
    # Predict and store next quarter GDP growth
    gp_pred = gp_model.predict(X_test, return_std = True)
    results.loc[first, ['GP', 'GP_std']] = [val[0] for val in gp_pred]
    
    ## BLR ###################################################################
    
    # Initializing ridge regression model
    blr_model = BayesianRidge(n_iter = 1000, normalize = True)
    
    # Fitting model
    blr_model.fit(X_train, y_train)
    
    # Predict and store next quarter GDP growth
    blr_pred = blr_model.predict(X_test, return_std = True)
    results.loc[first, ['BLR', 'BLR_std']] = [val[0] for val in blr_pred]
    
    ## E-NET #################################################################
    
    # Initializing elastic model
    en_model = ElasticNet(alpha=0.0010696120009825855, 
                          l1_ratio=0.5, 
                          max_iter = 2000, 
                          normalize = True)
    
    ## RIDGE #################################################################
    
    # Initializing ridge model
    ridge_model =  Ridge(alpha = 1.0,
                         normalize = True,
                         max_iter = 2000,
                         solver= 'auto')
    
    ## Bootstrapping Standard Errors for Ridge and EN ########################
    
    # Bootstrap predictions and standard errors and store next quarter GDP growth
    
    # number of samples
    n = 500
    samples = pd.DataFrame({'ridge' : np.empty(n), 'en' : np.empty(n)})
    
    for j in range(n):
        X_boot, y_boot = resample(X_train, y_train, n_samples = X_train.shape[1])
        ridge_model.fit(X_boot, y_boot)
        en_model.fit(X_boot, y_boot)
        samples.loc[j, 'ridge'] = ridge_model.predict(X_test)[0]
        samples.loc[j, 'en'] = en_model.predict(X_test)[0]    

    # Predict and store next quarter GDP growth
    results.loc[first, ['Ridge', 'Ridge_std']] = [np.mean(samples.ridge), np.std(samples.ridge)]
    results.loc[first, ['EN', 'EN_std']] = [np.mean(samples.en), np.std(samples.en)]

    # update index
    first += 1
    print(last - first)

# Results
print("MSE:")
for model in models:
    mse = mean_squared_error(results.loc[:,"Actual"], results.loc[:,model])
    print(model + ": " + str(mse))
print("Mean: " + str(np.mean((results.loc[:,"Actual"] - np.mean(results.loc[:,"Actual"]))**2)))
print("\n")

print("RMSE:")
for model in models:
    rmse = np.sqrt(mean_squared_error(results.loc[:,"Actual"], results.loc[:,model]))
    print(model + ": " + str(rmse))
    
print("\n")

print("MAE:")
for model in models:
    mae = mean_absolute_error(results.loc[:,"Actual"], results.loc[:,model])
    print(model + ": " + str(mae))
print("Mean: " + str(np.mean(np.abs(results.loc[:,"Actual"] - np.mean(results.loc[:,"Actual"])))))