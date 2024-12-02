import pandas as pd
import numpy as np

# 1. Leitura do Dataset
url = "autism_screening.csv"
data = pd.read_csv(url, header=0)

print(data.head())

data = data.drop(columns=['relation', 'age_desc', 'used_app_before'])
data = pd.get_dummies(data, columns=['austim'])

print(data.head())

