# First, let's load a portion of wine data set (columns 1, 2, and 9)
import pandas as pd
import numpy as np
%matplotlib inline

dataset = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None,
                      usecols=[1,2,9])

dataset.columns=['Alcohol', 'Malic acid', 'Color Intensity']

dataset.head()

# Let's plot our features to visualize their scale.
dataset.plot();

# For both types of scaling, we need to import preprocessing module
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dataset_standard_scaled = scaler.fit(dataset)
dataset_standard_scaled = dataset_standard_scaled.transform(dataset)

dataset_standard_scaled[:5]

# Let's visualize our scaled features (left) and their distribution (right)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,2)

pd.DataFrame(dataset_standard_scaled).plot(kind='line', legend=False, ax=axs[0], figsize=(15, 5))
pd.DataFrame(dataset_standard_scaled).plot(kind='hist', legend=False, ax=axs[1]);

print('mean = ' + str(abs(dataset_standard_scaled.mean().round())) + ' std = ' + str(abs(dataset_standard_scaled.std().round())))

# For both types of scaling, we need to import preprocessing module
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_min_max_scaled = scaler.fit(dataset)
dataset_min_max_scaled = dataset_min_max_scaled.transform(dataset)

dataset_min_max_scaled[:5]

# Let's visualize our scaled features
pd.DataFrame(dataset_min_max_scaled).plot(legend=False, figsize=(7, 5));

print('min = ' + str(np.min(dataset_min_max_scaled)) + ' max = ' + str(np.max(dataset_min_max_scaled)))
