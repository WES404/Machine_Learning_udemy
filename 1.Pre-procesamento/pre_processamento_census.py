import pandas as pd

base = pd.read_csv("census.csv")

previsores = base.iloc[:, 0:14].values

classe = base.iloc[:, 14].values

# Mudando dados categóricos para Númericos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelenconder_previsores = LabelEncoder()


onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')

previsores = onehotencoder.fit_transform(previsores).toarray()

labelenconder_classes = LabelEncoder()
classe = labelenconder_classes.fit_transform(classe)

# Escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
