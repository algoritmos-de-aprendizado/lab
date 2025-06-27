import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

df = pd.read_csv("./dataset.csv")

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

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

# ======= Treinamento para cada arquitetura =======
joblib.dump(vectorizer, "vectorizer_tfidf.pkl")

for num_layers in [1, 3, 5]:
    print(f"\n=== Treinando MLP com {num_layers} camada(s) oculta(s) ===")
    model = MLP(input_size=X_train.shape[1], hidden_size=128, num_classes=4, num_hidden_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(f"Loss por Época - {num_layers} camada(s)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/loss_{num_layers}_camadas.png")
    plt.close()

    # Avaliação
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(batch_y.numpy())

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

    torch.save(model.state_dict(), f"results/modelo_mlp_{num_layers}_camadas.pt")
