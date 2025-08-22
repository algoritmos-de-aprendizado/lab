import pandas as pd
import numpy as np
import itertools
import random

# ============================================================
# Laboratório: Conjuntos de Dados, Dados Estruturados e Não Estruturados,
# Espaço de Instâncias
# ============================================================
# Objetivo: implementar os exercícios propostos.
# ============================================================

# ------------------------------------------------------------
# Prática 1: Criar e manipular dados tabulares
# ------------------------------------------------------------
nomes = ["Ana", "Bruno", "Carla", "Daniel", "Elisa"]
idades = [23, 31, 19, 45, 28]
cidades = ["Curitiba", "São Paulo", "Belo Horizonte", "Rio", "Recife"]

df_pessoas = pd.DataFrame({"nome": nomes, "idade": idades, "cidade": cidades})
df_pessoas.to_csv("pessoas.csv", index=False)

df_lido = pd.read_csv("pessoas.csv")
print("Pessoas com idade > 25:")
print(df_lido[df_lido["idade"] > 25])

# ------------------------------------------------------------
# Prática 2: Geração de dados tabulares com distribuições
# ------------------------------------------------------------
altura = np.random.normal(170, 10, 100)
peso = np.random.normal(70, 15, 100)
classe = np.random.choice(["A", "B"], size=100)

df_dados = pd.DataFrame({"altura": altura, "peso": peso, "classe": classe})
df_dados.to_csv("dados.csv", index=False)

print("Médias por classe:")
print(df_dados.groupby("classe")[["altura", "peso"]])
print(df_dados.groupby("classe")[["altura", "peso"]].mean())

# ------------------------------------------------------------
# Prática 3: Espaço de instâncias discreto
# ------------------------------------------------------------
cores = ["vermelho", "verde", "azul"]
tamanhos = list(range(1, 11))
formas = ["círculo", "quadrado"]

instancias = list(itertools.product(cores, tamanhos, formas))
print(instancias)
df_instancias = pd.DataFrame(instancias, columns=["cor", "tamanho", "forma"])
df_instancias.to_csv("espaco_instancias.csv", index=False)

print("Número total de instâncias possíveis:", len(df_instancias))

# ------------------------------------------------------------
# Prática 4: Integração
# ------------------------------------------------------------
idades = np.random.randint(20, 81, size=20)
pressao = np.round(np.random.uniform(10.0, 18.0, size=20), 1)
diagnosticos = [
    "dores leves",
    "pressão alta",
    "sem sintomas",
    "fadiga crônica",
    "dor de cabeça",
    "mal estar geral",
]

diag_aleatorios = [random.choice(diagnosticos) for _ in range(20)]

df_pacientes = pd.DataFrame({"idade": idades, "pressao": pressao, "diagnostico": diag_aleatorios})
df_pacientes.to_csv("pacientes.csv", index=False)

df_pacientes_lido = pd.read_csv("pacientes.csv")
print("Pacientes com pressão > 14.0:")
print(df_pacientes_lido[df_pacientes_lido["pressao"] > 14.0])
