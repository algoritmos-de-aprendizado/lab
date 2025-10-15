from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X, y = load_wine(return_X_y=True)  Troque por outro conj. de dados

Use StandardScaler() aqui

X_filtro = SelectKBest(score_func=f_classif, k=5).fit_transform(X_scaled, y)

# Feature selection (wrapper)
logreg = LogisticRegression(max_iter=5000, solver='lbfgs')
X_wrapper = RFE(logreg, n_features_to_select=5).fit_transform(X_scaled, y)

# Evaluate
acc_filtro = cross_val_score(logreg, X_filtro, y, cv=5).mean()
acc_wrapper = cross_val_score(logreg, X_wrapper, y, cv=5).mean()
print(acc_filtro, acc_wrapper)

