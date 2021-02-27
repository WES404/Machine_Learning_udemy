import pandas as pd

base = pd.read_csv("census.csv")

previsores = base.iloc[:, 0:14].values

classe = base.iloc[:, 14].values

# Mudando dados categóricos para Númericos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')

# Muda os previsores para númericos sem 'peso'
previsores = onehotencoder.fit_transform(previsores).toarray()

# Muda as classes para numericas
labelencoder_classes = LabelEncoder()
classe = labelencoder_classes.fit_transform(classe)

# Escalonamento 
from sklearn.preprocessing import StandardScaler
# Escalona pelo método padrão
Escalar = StandardScaler()
previsores = Escalar.fit_transform(previsores)

from sklearn.model_selection import train_test_split
# Dividindo
X_treino, X_teste, y_treino, y_teste = train_test_split(previsores, classe, test_size=.33, random_state=0)

# Usando KNN
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors = 5, p = 2, metric= 'minkowski')
classificador.fit(X_treino, y_treino)

# Pevendo
previssoes = classificador.predict(X_teste)

# Checando presição
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(y_teste, previssoes)
matriz = confusion_matrix(y_teste, previssoes)