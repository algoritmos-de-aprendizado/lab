#  {Código de Apoio A – Definição de MLP simples com uma camada oculta}
# #################################
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),  # camada oculta com 4 neurônios
            nn.Tanh(),
            nn.Linear(4, 1),  # camada de saída
            nn.Sigmoid()  # saída para classificação binária
        )

    def forward(self, x):
        return self.model(x)


# Dados XOR
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

model = SimpleMLP()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Treinamento
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'Época {epoch}, Loss: {loss.item()}')

# Teste final
with torch.no_grad():
    predictions = model(X)
    print('\nPredições:')
    print(predictions)
