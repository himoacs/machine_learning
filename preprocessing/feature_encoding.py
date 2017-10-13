#importing pandas
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'/Users/himanshugupta/Documents/iris.csv')

# Let's view first few rows
dataset.head()

# Let's only take 10 random rows
dataset = dataset.sample(10)

# Let's see different species we have and count per species
dataset.groupby('species').count()

# Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
species = dataset['species']
species_encoded = encoder.fit_transform(species)
species_encoded

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
species_one_hot = encoder.fit_transform(species_encoded.reshape(-1,1))
species_one_hot.toarray()[:5]

# Encoding and One Hot Encoding combined
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
species_one_hot = encoder.fit_transform(dataset.species)

species_one_hot
