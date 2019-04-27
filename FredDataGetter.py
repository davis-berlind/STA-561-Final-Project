from fred import Fred
import numpy as np
import pandas as pd
from datetime import datetime
from quarterize import quarterMean, growthRate

class DataGetter:
    """
    Class capable of collecting and cleaning data from FRED API given a series code.

    Parameters:
        code (str): FRED code for data series of interest.
        dates: Series of dates marking the start of each quarter.
        method (bool): Method for aggregating data. level data (True) or percentage data (False). 
        params: FRED API parameters.
    """
    
    # determines whether the data series will be included. Defaults to true.
    include = True
    
    def __init__(self, code, dates, method, params, key):
        self.fr = Fred(api_key = key, response_type = 'df')
        self.code = code
        self.dates = dates
        self.method = method
        self.params = params
        self.start = ''
        self.key = key

    def levelShrink(self, dates, series):
        """ 
        Method for aggregating level data to same frequency as that of self.dates
        
        Returns:
            list of data aggregated to same frequency as self.dates
        """

        ret = []
        for start, end in zip(dates[:-1], dates[1:]):
            # get chunk of data between start and end dates
            batch = series.loc[(series.date >= start) & (series.date < end), 'value']
            
            # test if there are periods of missing values in series
            if (batch.isnull().all()):
                self.include = False  # don't include missing data
                break
            
            # test if there are periods of missing values or zeroes in series
            if (all((x == 0) | np.isnan(x) for x in batch)):
                self.include = False  # don't include missing data
                break
            # calculate within period percent change
            shrink = np.prod(1 + batch) - 1
            ret.append(shrink)
        return ret

    def percentShrink(self, dates, series):
        """ 
        Method for aggregating percentage data to same frequency as that of self.dates

        Returns:
            list of data aggregated to same frequency as self.dates
        """

        ret = []
        for start, end in zip(dates[:-1], dates[1:]):
            # get chunk of data between start and end dates
            batch = series.loc[(series.date >= start) & (series.date < end), 'value']
            
            # test if there are periods of missing values in series
            if (batch.isnull().all()):
                # don't include missing data
                self.include = False
                break
            
            # test if there are periods of missing values or zeroes in series
            if (all((x == 0) | np.isnan(x) for x in batch)):
                self.include = False  # don't include missing data
                break
            # calculate within period percent change
            shrink = np.sum(np.diff(batch))
            ret.append(shrink)
        return ret

    def getter(self):
        """ 
        Method for collecting data for code provided at initialization.
        Returns:
            Pandas Series of percent changes for self.code at same frequency as self.dates
        """
        self.start = self.fr.series.details(self.code).observation_start[0]
        series = self.fr.series.observations(self.code, params = self.params)
        
        # drop NAs in high frequency data
        series.dropna(axis = 0, inplace = True)
        
        # aggregate data and calculate percent changes
        if (self.method):
            series = self.percentShrink(self.dates, series)
        else:
            series = pd.DataFrame({'date' : series.date[1:], 'value': growthRate(series.value)})
            series = self.levelShrink(self.dates, series)
        return series