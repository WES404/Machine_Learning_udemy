import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]

# apagar dados 

base.drop('age', 1, inplace=True) # apaga a coluna inteira

base.drop(base[base.age < 0 ].index, inplace=True) # Apaga as linhas

# preencher valores com a média

base['age'].mean()
base.loc[base.age < 0, 'age'] = 40.92


# Valores nulos

pd.isnull(base['age'])

base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classes = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
# Muda os valores NaN pela média
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

from sklearn.preprocessing import StandardScaler
# Escalona pelo método padrão
Escalar = StandardScaler()
previsores = Escalar.fit_transform(previsores)
