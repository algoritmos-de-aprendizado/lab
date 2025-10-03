import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Dataset real (Breast Cancer) com atributos em escalas diferentes
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN sem transformação
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_original = accuracy_score(y_test, y_pred)

# KNN com normalização (MinMax)
scaler_minmax =
X_train_minmax =
X_test_minmax =

knn.fit(X_train_minmax, y_train)
y_pred_minmax = knn.predict(X_test_minmax)
acc_minmax = accuracy_score(y_test, y_pred_minmax)

# KNN com padronização (Z-score)
scaler_std =
X_train_std =
X_test_std =

knn.fit(X_train_std, y_train)
y_pred_std = knn.predict(X_test_std)
acc_std = accuracy_score(y_test, y_pred_std)

# Mostrar comparação em gráfico
plt.figure(figsize=(6, 4))
accs = [acc_original, acc_minmax, acc_std]
labels = ["Sem Escala", "Normalização (MinMax)", "Padronização (Z-score)"]

sns.barplot(x=labels, y=accs)
plt.ylim(0.8, 1.0)
plt.ylabel("Acurácia no teste")
plt.title("Impacto da Escala de Dados no KNN (Breast Cancer)")
plt.show()

print("Acurácia sem escala:", acc_original)
print("Acurácia com normalização:", acc_minmax)
print("Acurácia com padronização:", acc_std)

##########################################################

# Gráficos comparativos
plt.figure(figsize=(15, 5))
# Original vs Normalizado
plt.subplot(1, 2, 1)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], alpha=0.5, label="Original")
plt.scatter(X_train_minmax[:, 0], X_train_minmax[:, 1], alpha=0.5, label="Normalizado (MinMax)")
plt.xlabel("mean radius")
plt.ylabel("worst area")
plt.legend()
plt.title("Comparação Original vs Normalizado")

# Original vs Padronizado
plt.subplot(1, 2, 2)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], alpha=0.5, label="Original")
plt.scatter(X_train_std[:, 0], X_train_std[:, 1], alpha=0.5, label="Padronizado (Z-score)")
plt.xlabel("mean radius")
plt.ylabel("worst area")
plt.legend()
plt.title("Comparação Original vs Padronizado")

plt.tight_layout()
plt.show()
