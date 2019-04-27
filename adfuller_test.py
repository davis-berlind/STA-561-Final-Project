from statsmodels.tsa.stattools import adfuller
import pandas as pd

X = pd.read_csv("data.csv").drop('date', axis = 1)

# p-value <= 0.05: Reject the null hypothesis (H0) 
# the data does not have a unit root and is stationary.

adfuller_results = X.apply(adfuller, axis = 1)
failed = 0
p_val = 0.05
for i, result in zip(range(X.shape[1]), adfuller_results):
    if (result[1] > p_val): 
        failed += 1
        print(i)
print(failed)