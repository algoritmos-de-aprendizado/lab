# -*- coding: utf-8 -*-
"""
Laboratório: Dependência entre Atributos
Objetivo: Analisar relações entre variáveis em conjuntos de dados
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
sns.set_palette("husl")

print("=== LABORATÓRIO: DEPENDÊNCIA ENTRE ATRIBUTOS ===\n")

# 1. CARREGANDO E EXPLORANDO OS DADOS
print("1. CARREGANDO E EXPLORANDO OS DADOS")

# Dataset Iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names_iris = iris.feature_names

print(f"Dataset Iris - Forma: {X_iris.shape}")
print(f"Atributos: {feature_names_iris}")
print(f"Classes: {iris.target_names}")

# Dataset Wine
wine = load_wine()
X_wine = wine.data
y_wine = wine.target
feature_names_wine = wine.feature_names

print(f"\nDataset Wine - Forma: {X_wine.shape}")
print(f"Número de atributos: {len(feature_names_wine)}")

# 2. ANÁLISE DE CORRELAÇÃO
print("\n2. ANÁLISE DE CORRELAÇÃO")


def analisar_correlacoes(X, feature_names, dataset_name):
    """Analisa correlações entre atributos"""
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr()

    # Correlação de Pearson
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, fmt='.2f')
    plt.title(f'Matriz de Correlação - {dataset_name}')
    plt.tight_layout()
    plt.show()

    # Encontrar pares mais correlacionados
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature1': corr_matrix.columns[i],
                'Feature2': corr_matrix.columns[j],
                'Correlação': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Abs_Corr'] = abs(corr_df['Correlação'])
    top_corr = corr_df.nlargest(5, 'Abs_Corr')

    print(f"\nTop 5 pares mais correlacionados ({dataset_name}):")
    for _, row in top_corr.iterrows():
        print(f"  {row['Feature1']} - {row['Feature2']}: {row['Correlação']:.3f}")

    return corr_matrix


# Análise para Iris
corr_iris = analisar_correlacoes(X_iris, feature_names_iris, "Iris Dataset")

# Análise para Wine (apenas primeiras 8 features para visualização)
corr_wine = analisar_correlacoes(X_wine[:, :8], feature_names_wine[:8], "Wine Dataset (primeiras 8 features)")

# 3. INFORMAÇÃO MÚTUA
print("\n3. INFORMAÇÃO MÚTUA")


def analisar_informacao_mutua(X, y, feature_names, dataset_name):
    """Calcula informação mútua entre features e target"""
    mi = mutual_info_classif(X, y, random_state=42)

    plt.figure(figsize=(10, 6))
    indices = np.argsort(mi)[::-1]
    plt.bar(range(len(mi)), mi[indices])
    plt.xticks(range(len(mi)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Informação Mútua com Target - {dataset_name}')
    plt.ylabel('Informação Mútua')
    plt.tight_layout()
    plt.show()

    mi_df = pd.DataFrame({
        'Feature': feature_names,
        'MI_Score': mi
    }).sort_values('MI_Score', ascending=False)

    print(f"\nInformação Mútua com Target ({dataset_name}):")
    for _, row in mi_df.head().iterrows():
        print(f"  {row['Feature']}: {row['MI_Score']:.4f}")

    return mi_df


# Análise de informação mútua
mi_iris = analisar_informacao_mutua(X_iris, y_iris, feature_names_iris, "Iris")
mi_wine = analisar_informacao_mutua(X_wine, y_wine, feature_names_wine, "Wine")

# 4. IMPORTÂNCIA DE ATRIBUTOS COM RANDOM FOREST
print("\n4. IMPORTÂNCIA DE ATRIBUTOS COM RANDOM FOREST")


def analisar_importancia_rf(X, y, feature_names, dataset_name):
    """Analisa importância de features usando Random Forest"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importancia = rf.feature_importances_
    indices = np.argsort(importancia)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importancia)), importancia[indices])
    plt.xticks(range(len(importancia)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Importância de Features (Random Forest) - {dataset_name}')
    plt.ylabel('Importância')
    plt.tight_layout()
    plt.show()

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importancia
    }).sort_values('Importance', ascending=False)

    print(f"\nImportância de Features - Random Forest ({dataset_name}):")
    for _, row in feat_imp_df.head().iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    return feat_imp_df


# Análise de importância
rf_iris = analisar_importancia_rf(X_iris, y_iris, feature_names_iris, "Iris")
rf_wine = analisar_importancia_rf(X_wine, y_wine, feature_names_wine, "Wine")

# 5. DETECÇÃO DE MULTICOLINEARIDADE
print("\n5. DETECÇÃO DE MULTICOLINEARIDADE")


def analisar_multicolinearidade(X, feature_names, dataset_name):
    """Analisa multicolinearidade usando VIF (Fator de Inflação de Variância).

    Usa `statsmodels` se disponível; caso contrário, cai para uma implementação
    de fallback que calcula VIF usando regressão linear do `sklearn`.
    """
    # Tentar usar statsmodels (mais preciso / padrão) e, se não estiver
    # disponível, usar um fallback com sklearn
    try:
        # Import statsmodels dynamically to avoid static analysis/import-time errors
        import importlib

        sm_outliers = importlib.import_module('statsmodels.stats.outliers_influence')
        variance_inflation_factor = getattr(sm_outliers, 'variance_inflation_factor')
        sm_tools = importlib.import_module('statsmodels.tools.tools')
        add_constant = getattr(sm_tools, 'add_constant')

        # Adicionar constante para cálculo do VIF
        X_with_const = add_constant(X)

        # Calcular VIF para cada feature (pular a constante)
        vif_data = []
        for i in range(1, X_with_const.shape[1]):
            vif = variance_inflation_factor(X_with_const, i)
            vif_data.append({'Feature': feature_names[i - 1], 'VIF': vif})

        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

        print(f"\nAnálise de Multicolinearidade - VIF ({dataset_name})")
        print("(usando statsmodels)")
    except Exception:
        # Fallback: calcular VIF usando regressões com sklearn
        print(f"\nstatsmodels não encontrado — usando fallback baseado em sklearn para VIF ({dataset_name}).")
        from sklearn.linear_model import LinearRegression

        X_arr = np.asarray(X, dtype=float)
        n_features = X_arr.shape[1]
        vif_vals = []

        for i in range(n_features):
            # y = feature i, X_other = todas as outras features
            y = X_arr[:, i]
            X_other = np.delete(X_arr, i, axis=1)

            # Ajustar regressão linear e calcular R^2
            reg = LinearRegression()
            reg.fit(X_other, y)
            r2 = reg.score(X_other, y)

            # Evitar divisão por zero/valores numéricos inválidos
            if r2 >= 1.0:
                vif = float('inf')
            else:
                vif = 1.0 / (1.0 - r2)

            vif_vals.append(vif)

        vif_data = [{'Feature': feature_names[i], 'VIF': vif_vals[i]} for i in range(n_features)]
        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

        print("(usando fallback sklearn)")

    print("VIF > 10 indica multicolinearidade problemática")
    for _, row in vif_df.iterrows():
        status = "⚠️ ALTO" if row['VIF'] > 10 else "✅ OK"
        # Tratar inf/NaN ao imprimir
        try:
            vif_str = f"{row['VIF']:.2f}"
        except Exception:
            vif_str = str(row['VIF'])
        print(f"  {row['Feature']}: {vif_str} {status}")

    return vif_df


# Análise de multicolinearidade
vif_iris = analisar_multicolinearidade(X_iris, feature_names_iris, "Iris")
vif_wine = analisar_multicolinearidade(X_wine, feature_names_wine, "Wine")

# 6. VISUALIZAÇÃO DE RELAÇÕES ENTRE ATRIBUTOS
print("\n6. VISUALIZAÇÃO DE RELAÇÕES ENTRE ATRIBUTOS")


def visualizar_relacoes(X, y, feature_names, target_names, dataset_name):
    """Visualiza relações entre pares de atributos"""
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Selecionar os 4 atributos mais importantes
    mi = mutual_info_classif(X, y, random_state=42)
    top_features_idx = np.argsort(mi)[-4:][::-1]
    top_features = [feature_names[i] for i in top_features_idx]

    # Pairplot com os atributos mais importantes
    plot_df = df[top_features + ['target']]

    plt.figure(figsize=(12, 10))
    sns.pairplot(plot_df, hue='target', palette='viridis',
                 diag_kind='hist', corner=False)
    plt.suptitle(f'Relações entre Atributos - {dataset_name}', y=1.02)
    plt.show()

    # Scatter plot dos dois atributos mais importantes
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, top_features_idx[0]], X[:, top_features_idx[1]],
                          c=y, cmap='viridis', alpha=0.7)
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Relação entre os 2 atributos mais importantes - {dataset_name}')
    plt.colorbar(scatter)
    plt.show()


# Visualizações
visualizar_relacoes(X_iris, y_iris, feature_names_iris, iris.target_names, "Iris")
visualizar_relacoes(X_wine, y_wine, feature_names_wine, wine.target_names, "Wine")

# 7. ANÁLISE DE DEPENDÊNCIA LINEAR VS NÃO-LINEAR
print("\n7. ANÁLISE DE DEPENDÊNCIA LINEAR VS NÃO-LINEAR")


def comparar_dependencias(X, feature_names, dataset_name):
    """Compara dependências lineares e não-lineares"""
    n_features = min(4, X.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i in range(n_features):
        for j in range(i + 1, min(i + 3, X.shape[1])):
            if (i * 2 + j - i - 1) < len(axes):
                ax = axes[i * 2 + j - i - 1]

                # Correlação linear (Pearson)
                corr_pearson, _ = pearsonr(X[:, i], X[:, j])

                # Correlação não-linear (Spearman)
                corr_spearman, _ = spearmanr(X[:, i], X[:, j])

                ax.scatter(X[:, i], X[:, j], alpha=0.6)
                ax.set_xlabel(feature_names[i])
                ax.set_ylabel(feature_names[j])
                ax.set_title(f'{feature_names[i]} vs {feature_names[j]}\n'
                             f'Pearson: {corr_pearson:.3f}, Spearman: {corr_spearman:.3f}')

    plt.tight_layout()
    plt.suptitle(f'Comparação Dependências Lineares vs Não-lineares - {dataset_name}',
                 y=1.02, fontsize=14)
    plt.show()


# Comparação de dependências
comparar_dependencias(X_iris, feature_names_iris, "Iris")
comparar_dependencias(X_wine[:, :6], feature_names_wine[:6], "Wine (6 primeiras features)")

print("\n=== ANÁLISE CONCLUÍDA ===")
print("\nResumo dos principais achados:")
print("1. Correlação mede relações lineares entre atributos")
print("2. Informação Mútua captura dependências lineares e não-lineares")
print("3. Random Forest fornece importância baseada em poder preditivo")
print("4. VIF ajuda a identificar multicolinearidade problemática")
print("5. Dependências não-lineares podem ser capturadas por Spearman")
