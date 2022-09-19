import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# %matplotlib inline

# Predicción de la calidad del vino tinto utilizando Random Forest
df = pd.read_csv (r"C:\Users\anton\OneDrive\Documents/winequality-red.csv")   
df.head()
# Descripción de los datos
df.describe()
# Conteo de los registros de calidad del vino
print(df['quality'].value_counts())
# Gráfica registros de la calidad del vino
plt.figure(1, figsize=(10,10))
df['quality'].value_counts().plot.pie(autopct="%1.2f%%")
X = df.drop('quality', axis = 1)
y = df.quality             # Variable dependiente
# Separando Conjunto de datos 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 42)
kf = KFold(n_splits=5)
# Creando Modelo 
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
score = rfc.score(X_train,y_train)
print("Metrica del modelo", score)
scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring="accuracy")
print("Metricas cross_validation", scores)
print("Media de cross_validation", scores.mean())

# Predicción Calidad del Vino
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))

score_pred = metrics.accuracy_score(y_test, pred_rfc)
print("Metrica en Test", score_pred)

##Evaluación del desempeño
print("Debido a los errores obtenidos del modelo, se puede decir que existe un bajo BIAS, lo que sugiere menos suposiciones sobre la forma de la función objetivo")
print("Se tiene alta varianza, lo que sugiere grandes cambios en la estimación de la función objetivo, con cambios en el conjunto de datos de capacitación")
print("El modelo se encuentra en underfitting, lo que explica su incapacidad para obtener resultados correctos debido al entrenamiento que tuvo")


# Basándote en lo encontrado en tu análisis utiliza técnicas de regularización o ajuste de parámetros para mejorar el desempeño de tu modelo y documenta en tu reporte cómo mejoró este.
print("Si reducimos la proporción de nuestros conjuntos de datos a 80/20 y modificamos los las muestras mezcladas incrementando nuestro parámetro random_state 60 y el número de estimadores para el random forest a 230, nos encontramos con una ligera mejora dentro del modelo")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 60)
kf = KFold(n_splits=5)
# Creando Modelo 
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
score = rfc.score(X_train,y_train)
print("Metrica del modelo", score)
scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring="accuracy")
print("Metricas cross_validation", scores)
print("Media de cross_validation", scores.mean())

# Predicción Calidad del Vino
rfc = RandomForestClassifier(n_estimators=230)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))

score_pred = metrics.accuracy_score(y_test, pred_rfc)
print("Metrica en Test", score_pred)