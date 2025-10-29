from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

plt.figure(figsize=(15, 10))
plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Árvore de Decisão - Dataset Iris')
plt.show()
