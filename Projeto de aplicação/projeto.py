import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Leitura do Dataset
url = "Projeto de aplicação/autism_screening.csv"
data = pd.read_csv(url, header=0)

# 2. Processamento dos dados
data = data.drop(columns=['relation', 'age_desc', 'used_app_before'])
data = data.rename(columns={'austim': 'autism_family', 'Class/ASD': 'autism'})
data = pd.get_dummies(data, columns=['autism_family', 'gender', 'jundice', 'autism'])