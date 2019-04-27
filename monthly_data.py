from fred import Fred
import numpy as np
import pandas as pd
from datetime import datetime
import time
from quarterize import quarterMean, growthRate
from dateutil.relativedelta import relativedelta

# FRED API key
# key = 'enter your key'
%run -i fred_api_key

## Getting Data from FRED ##

# loading keys
fseries = pd.read_csv('FRED_data_series.csv')
data = pd.read_csv('data.csv')

# Intialize Fred object for communicating with FRED API
fr = Fred(api_key = key, response_type = 'df')

start = datetime(1990, 1, 1)
end = datetime(2018, 10, 2)   

params = {'observation_start': start.strftime('%Y-%m-%d'),
          'observation_end': end.strftime('%Y-%m-%d')}

dates = []

while start <= end:
    dates.append(start.strftime('%Y-%m-%d'))
    start += relativedelta(months = 1)

# Retrieving data from Fred

# number of series to collect
n = fseries.shape[0]

# initialize data frame for data collection
monthly_data = pd.DataFrame({'date' : dates[1:]})

# loop through and collect data for each code in fseries
i = 0
for col in data.columns[1:-4]:
	method = int(fseries.loc[fseries.file == col, 'method'])
	freq = fr.series.details(col).frequency_short[0]
	series = fr.series.observations(col, params = params)

	if ((freq == 'M') & bool(method)):
		monthly_data[col] = np.diff(series.value)
	elif(freq == 'M'):
		monthly_data[col] = growthRate(series.value + 0.0001)
	elif(bool(method)):
		series.dropna(axis = 0, inplace = True)
		ret = []
		for start, end in zip(dates[:-1], dates[1:]):
			# get chunk of data between start and end dates
			batch = series.loc[(series.date >= start) & (series.date < end), 'value']

			# calculate within period percent change
			shrink = np.sum(np.diff(batch))
			ret.append(shrink)
		monthly_data[col] = ret
	else:
		series.dropna(axis = 0, inplace = True)
		series =  pd.DataFrame({'date' : series.date[1:], 'value': growthRate(series.value + 0.0001)})
		ret = []
		for start, end in zip(dates[:-1], dates[1:]):
			# get chunk of data between start and end dates
			batch = series.loc[(series.date >= start) & (series.date < end), 'value']

			# calculate within period percent change
			shrink = np.prod(1 + batch) - 1
			ret.append(shrink)
		monthly_data[col] = ret
	i+=1
	if((i % 100) == 0): print(i)

	# time buffer to prevent 429 server error
	time.sleep(.5)

# adding in zero-indexed data
data['MEIM683SFRBCHI'] = fr.series.observations('MEIM683SFRBCHI', params = params).value[1:]
data['RMEIM683SFRBCHI'] = fr.series.observations('RMEIM683SFRBCHI', params = params).value[1:]

# adding lagged GDP
lag1 = {'observation_start': "1989-10-01",
        'observation_end': "2018-07-01"}

lag2 = {'observation_start': "1989-07-01",
        'observation_end': "2018-04-01"}
