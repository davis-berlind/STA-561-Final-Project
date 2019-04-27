from dateutil.relativedelta import relativedelta
monthly_data = pd.read_csv('monthly.csv')

one_month = monthly_data.iloc[:-1,:]
two_month = monthly_data.iloc[:-2,:]

for col in monthly_data.columns[1:]:
    
    # Initializing AR(2,0) model
    val1 = one_month[col].values
    val1 = val1[~np.isnan(val1)]
    val2 = one_month[col].values
    val2 = val2[~np.isnan(val2)]
    
    ar1 = ARMA(val1, order = (2,0))
    ar2 = ARMA(val2, order = (2,0))
    
    # Fitting model
    ar1_model = ar1.fit(transparams = False, disp = False)
    ar2_model = ar2.fit(transparams = False, disp = False)

    # Storing Values
    one_month.loc[-1, col] = ar1_model.forecast(1)[0][0]
    two_month.loc[-2, col] = ar2_model.forecast(2)[0][0]
    two_month.loc[-1, col] = ar2_model.forecast(2)[0][1]

one_month.date = monthly_data.date
two_month.date = monthly_data.date

def levelShrink(dates, series):
    ret = []
    for start, end in zip(dates[:-1], dates[1:]):
        # get chunk of data between start and end dates
        batch = series.loc[(series.date >= start) & (series.date < end), 'value']
            
        # calculate within period percent change
        shrink = np.prod(1 + batch) - 1
        ret.append(shrink)
    return ret

data1 = pd.read_csv('data.csv')
data2 = data1.copy()
fseries = pd.read_csv('FRED_data_series.csv')

start = datetime(1990, 1, 1)
end = datetime(2018, 10, 2)   

dates = []

while start <= end:
    dates.append(start.strftime('%Y-%m-%d'))
    start += relativedelta(months = 3)
    
for col in data1.columns[1:-4]:
    method = int(fseries.loc[fseries.file == col, 'method'])
    if (bool(method)):
        ret1 = [one_month[col].values[-1], data[col].values[-2], data[col].values[-3]]
        ret1 = np.sum(np.diff(ret1))
        ret2 = [two_month[col].values[-2], two_month[col].values[-1], data[col].values[-3]]
        ret2 = np.sum(np.diff(ret2))
    else:
        ret1 = [one_month[col].values[-1], data[col].values[-2], data[col].values[-3]]
        ret1 = np.asarray(ret1)
        ret1 = np.prod(1 + ret1) - 1
        ret2 = [two_month[col].values[-2], two_month[col].values[-1], data[col].values[-3]]
        ret2 = np.asarray(ret2)
        ret2 = np.prod(1 + ret2) - 1
            
    data1.loc[data.shape[0] - 1, col] = ret1
    data2.loc[data.shape[0] - 1, col] = ret2

other = ['MEIM683SFRBCHI', 'RMEIM683SFRBCHI']
for col in other:
    ret1 = [one_month[col].values[-1], data[col].values[-2], data[col].values[-3]]
    ret1 = np.mean(ret1)
    ret2 = [two_month[col].values[-2], two_month[col].values[-1], data[col].values[-3]]
    ret2 = np.mean(ret2)
    
    data1.loc[data.shape[0] - 1, col] = ret1
    data2.loc[data.shape[0] - 1, col] = ret2

data1.to_csv('one_month.csv', index = False)
data2.to_csv('two_month.csv', index = False)