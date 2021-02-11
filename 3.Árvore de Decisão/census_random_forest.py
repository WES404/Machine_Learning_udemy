import pandas as pd

base = pd.read_csv("census.csv")

previsores = base.iloc[:, 0:14].values

classe = base.iloc[:, 14].values

# Mudando dados categóricos para Númericos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelenconder_previsores = LabelEncoder()

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')

# Muda os previsores para númericos sem 'peso'
previsores = onehotencoder.fit_transform(previsores).toarray()

# Muda as classes para numericas
labelenconder_classes = LabelEncoder()
classe = labelenconder_classes.fit_transform(classe)

from sklearn.model_selection import train_test_split
# Dividindo
X_treino, X_teste, y_treino, y_teste = train_test_split(previsores, classe, test_size=.33, random_state=0)

# Usando Random Forest
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classificador.fit(X_treino, y_treino)

# Pevendo
previssoes = classificador.predict(X_teste)

# Checando presição
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(y_teste, previssoes)

matriz = confusion_matrix(y_teste, previssoes)