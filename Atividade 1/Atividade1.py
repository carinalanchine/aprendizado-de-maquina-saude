import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy import stats

# 1. Leitura do Dataset
url = "/heart.csv"
data = pd.read_csv(url, header=0)

cols_with_zeros = ['trestbps', 'chol', 'oldpeak', 'thalach']
data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)

cols_normalize=['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
data[cols_normalize]=(data[cols_normalize]-data[cols_normalize].min())/(data[cols_normalize].max()-data[cols_normalize].min())

print(data.head())

# 2. Análise Descritiva
print(data.describe())

# 3. Visualização dos Dados
for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        plt.figure(figsize=(8, 6))
        
        sns.histplot(data=data, x=column, hue='target', multiple='stack', bins=25, palette={0: 'blue', 1: 'red'}, alpha=0.6, edgecolor='black')
        
        plt.title(f'Histograma de {column}')
        plt.xlabel(column)
        plt.ylabel('Frequência')
        plt.legend(title='Doença Cardíaca', labels=['Positivo', 'Negativo'])
        
        plt.show()
        
sns.pairplot(data, hue='target', palette={0: 'blue', 1: 'red'})
plt.title('Gráfico de Dispersão de Todas as Variáveis')
plt.show() 

# 4. Teste de Correlação
cleaned_data = data[['chol', 'age', 'thalach']].dropna()

correlation, p_value = stats.pearsonr(cleaned_data['chol'], cleaned_data['age'])
print(f'Coeficiente de Correlação (age x chol): {correlation:.4f}, p-valor: {p_value:.4f}')

correlation, p_value = stats.pearsonr(cleaned_data['thalach'], cleaned_data['age'])
print(f'Coeficiente de Correlação (age x thalach): {correlation:.4f}, p-valor: {p_value:.4f}')

# 5. Regressão Logística
X = data[['age', 'sex', 'cp', 'chol', 'thalach', 'ca']].dropna()
y = data.loc[X.index, 'target']

model = LogisticRegression()
model.fit(X, y)

coef = pd.DataFrame(model.coef_, columns=X.columns)
coef['Intercept'] = model.intercept_
print(coef)

odds_ratios = np.exp(model.coef_[0])
print('Odds Ratios:')
for feature, ratio in zip(X.columns, odds_ratios):
    print(f'{feature}: {ratio:.4f}')