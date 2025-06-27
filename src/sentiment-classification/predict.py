import torch
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Carrega o vetor TF-IDF previamente salvo
vectorizer = joblib.load("vectorizer_tfidf.pkl")

# Define a arquitetura da MLP genérica (parametrizada por número de camadas)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_hidden_layers):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Frases para teste
novas_frases = [
    "meu wifi nao para de cair aqui",
    "preciso pagar meu boleto, com quem eu falo?",
    "cara minha internet ta muito lenta velho, sera que conectou no 2.4?",
    "quero cancelar",
]

# Transforma as frases com o vetor TF-IDF
X_new = vectorizer.transform(novas_frases).toarray()
X_tensor = torch.tensor(X_new, dtype=torch.float32)

CLASSES = {
    0: "Neutro / Calmo",
    1: "Levemente irritado",
    2: "Irritado",
    3: "Muito irritado"
}

# Testa os modelos treinados
for num_layers in [1, 3, 5]:
    model_path = f"results/modelo_mlp_{num_layers}_camadas.pt"
    print(f"\n=== Resultados com MLP de {num_layers} camada(s) oculta(s) ===")

    input_size = X_tensor.shape[1]
    model = MLP(input_size=input_size, hidden_size=128, num_classes=4, num_hidden_layers=num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)

    for frase, pred in zip(novas_frases, preds):
        print(f"Frase: '{frase}'\n → Classe prevista: {CLASSES[pred.item()]}\n")
