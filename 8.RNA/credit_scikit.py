import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')

# preencher valores com a média
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
# Muda os valores NaN pela média
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
# Escalona pelo método padrão
Escalar = StandardScaler()
previsores = Escalar.fit_transform(previsores)

# Separa o treinamento e o teste
from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(previsores, classe, test_size=.25, random_state=0)

# Usando rede neurais
from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose=True,
                              max_iter=1000,
                              tol=0.000001,
                              hidden_layer_sizes=(100),
                              activation="relu",)
classificador.fit(X_treino, y_treino)

# Prevendo
previssoes = classificador.predict(X_teste)

# Checando presição
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(y_teste, previssoes)
matriz = confusion_matrix(y_teste, previssoes)
