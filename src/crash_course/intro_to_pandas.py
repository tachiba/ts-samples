# https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=pandas-colab&hl=ja#scrollTo=aSRYu62xUi3g
import pandas as pd
import numpy as np

print(pd.__version__)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

print(city_names.index)

population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({'City name': city_names, 'Population': population})

print(cities.index)

cities.reindex([2, 0, 1])
cities.reindex(np.random.permutation(cities.index))

print(type(cities['City name']))
print(type(cities['City name'][1]))

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']

print(cities['Population density'])

cities['Saint 50'] = cities['City name'].apply(lambda name: name.startswith('Saint')) \
                     & cities['Area square miles'] > 50
print(cities['Saint 50'])

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())

hist = california_housing_dataframe.hist('housing_median_age')
