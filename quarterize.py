import numpy as np

def quarterMean(series, dates):
	"""
	Converts data at higher than quarterly frequency to quarterly data by
	imputing the within quarter mean of the data.

	Parameters:
		series: (2 x n) DataFrame of dates and data at higher than quarterly frequency.
		dates: Series of dates marking the start of each quarter.

	Returns:
		size n-1 list of within quarterly averages
	"""

	ret = []
	for start, end in zip(dates[:-1], dates[1:]):
		batch = series.loc[(series.date >= start) & (series.date < end), 'value']
		shrink = np.mean(batch)
		ret.append(shrink)
		return ret

def growthRate(series):
	"""
	Calculates growth rate of a data series.

	Parameters:
		series: length n numpy array of data

	Returns:
		length n-1 numpy array of growth rates
	"""
	
	return np.exp(np.diff(np.log(series))) - 1