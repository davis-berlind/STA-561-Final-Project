from fred import Fred
import numpy as np
import pandas as pd
from datetime import datetime
import time
from quarterize import quarterMean, growthRate
from FredDataGetter import DataGetter

# FRED API key
# key = 'enter your key'
%run -i fred_api_key

## Getting Data from FRED ##

# loading keys
fseries = pd.read_csv('FRED_data_series.csv')

# Intialize Fred object for communicating with FRED API
fr = Fred(api_key = key, response_type = 'df')

# Setting data start and end dates
start = datetime(1990, 1, 1)
end = datetime(2018, 10, 2)

params = {'observation_start': start.strftime('%Y-%m-%d'),
          'observation_end': end.strftime('%Y-%m-%d')}

# Downloading real GDP as the independent variable
gdp = fr.series.observations("GDPC1", params = params)
gdp.drop(labels = ['realtime_end', 'realtime_start'], axis = 1, inplace = True)
gdp.rename(columns = {'value' : 'gdp'}, inplace = True)
gdp.loc[1:, 'gdp'] = growthRate(gdp.gdp)
gdp = gdp.iloc[1:,:]

# Output GDP 
gdp.to_csv('gdp_growth.csv', index = False)

# Retrieving data from Fred

# number of series to collect
n = fseries.shape[0]

# fix dates to limits/frequency of GDP
dates = gdp.date

# initialize data frame for data collection
data = pd.DataFrame({'date' : dates[1:]})

# loop through and collect data for each code in fseries
for i in range(657,n):
    code = fseries.loc[i, 'file']
    method = fseries.loc[i, 'method']
    
    dg = DataGetter(code, dates, method, params, key)
    series = dg.getter()
    
    # only include series that start before specified start date and are "complete"
    if ((dg.start <= start) & dg.include):
        data[code] = series
    if (i % 100  == 0): 
        print(i)

    # time buffer to prevent 429 server error
    time.sleep(.5)

# adding in zero-indexed data
data['MEIM683SFRBCHI'] = quarterMean(fr.series.observations('MEIM683SFRBCHI', params = params), dates)
data['RMEIM683SFRBCHI'] = quarterMean(fr.series.observations('RMEIM683SFRBCHI', params = params), dates)

# adding lagged GDP
lag1 = {'observation_start': "1989-10-01",
        'observation_end': "2018-07-01"}

lag2 = {'observation_start': "1989-07-01",
        'observation_end': "2018-04-01"}

data['GDPLAG1'] = growthRate(fr.series.observations('GDPC1', params = lag1).value)
data['GDPLAG2'] = growthRate(fr.series.observations('GDPC1', params = lag2).value)

data.to_csv('data.csv', index = False)