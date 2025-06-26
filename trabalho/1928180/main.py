#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, adjusted_rand_score, silhouette_score, rand_score, accuracy_score
from mlxtend.plotting import plot_confusion_matrix

from tf_som import SOM
from label_and_test import label, test

# Hiperparâmetros
train_data = 60000
label_data = int(train_data*0.01) # 1% dos dados de treinamento para rotulação.
test_data = 10000

# MNIST contém dados de 28x28, que transformado fica 784 pixels, este é o dado de entrada.
input_dim = 784

width = 1 # Largura do mapa de saída.
height = 1 # Altura do mapa de saída.
epochs = 10
learning_rate_0 = 1.0
learning_rate_T = 0.01
neigbourhood_radius_0 = 10.0
neigbourhood_radius_T = 0.01

class_nbr = 10 # Número de classes sendo de 0 a 9.

# Importando o dataset.
(x_train_all, index_train_all), (x_test_all, index_test_all) = mnist.load_data()

# Normalização -> Como os pixels tem um range de 0 a 255, então é dividido por 255 para converter números de 0 a 1.
x_train_all = x_train_all.astype('float32') / 255.
x_test_all = x_test_all.astype('float32') / 255.

# Transformação da imagem em um vetor único de 28x28 = 784 pixels
x_train_all = x_train_all.reshape((60000, 784))
x_test_all = x_test_all.reshape((10000, 784))

# Construção do datasets de treino e dado
x_train = np.copy(x_train_all[:train_data,:])
x_label = np.copy(x_train_all[:label_data,:])
y_label = np.copy(index_train_all[:label_data])
x_test = np.copy(x_test_all[:test_data,:])
y_test = np.copy(index_test_all[:test_data])

# %%
print("\nHiperparâmetros configurados:" +
        f"\n - Taxa de aprendizado inicial = {learning_rate_0}" +
        f"\n - Taxa de aprendizado final = {learning_rate_T}" +
        f"\n - Raio inicial da vizinhança = {neigbourhood_radius_0}" +
        f"\n - Raio final da vizinhança = {neigbourhood_radius_T}")

#%%
# Etapa 1. Configurar a rede SOM.
som = SOM(
        width = width,
        height = height,
        input_dim = input_dim,
        initial_learning_rate = learning_rate_0,
        final_learning_rate = learning_rate_T,
        initial_neighbourhood_radius = neigbourhood_radius_0,
        final_neighbourhood_radius = neigbourhood_radius_T,
        epochs = epochs
)

# start_time = timeit.default_timer()
som.train(x_train)

#%%
# display neurons weights as mnist digits
def plot(weights):
    weights = weights.numpy()
    som_grid = plt.figure(figsize=(10, 10)) # width, height in inches
    for n in range(width*height):
        image = weights[n].reshape([28,28]) # x_train[num] is the 784 normalized pixel values
        sub = som_grid.add_subplot(width, height, n + 1)
        sub.set_axis_off()
        clr = sub.imshow(image, cmap = plt.get_cmap("jet"), interpolation = "nearest")

i = 0
for weight in som.saved_weights[0::10]:
    plot(weight)
    plt.savefig(f"plot_vid/som_weights_{i}.png", bbox_inches='tight', pad_inches=0)
    i = i + 1
#%%
# Aprendizado supervisionado - Rotulando os grupos
weights = som.get_weights().numpy()

# Rotulando a rede
neuron_label = label(class_nbr, weights, x_label, y_label)

# Salvando os rótulos em um arquivo txt
aux = neuron_label.reshape(width,height)
file_object = open(f'labels_{width}x{height}.txt', 'a')
for i in range(width):
    file_object.write(str(aux[i])+'\n')
file_object.close()

# Testar a rede
y_pred = test(weights, x_test, neuron_label)

# %%
# Métricas
print("Accuracy Score = ", accuracy_score(y_test, y_pred))

print("Adjusted Rand Score = ", adjusted_rand_score(y_test, y_pred))

print("Rand Score = ", rand_score(y_test, y_pred))

#%%
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                show_normed=True,
                                show_absolute=False,
                                class_names=[0,1,2,3,4,5,6,7,8,9],
                                figsize=(8, 8))

fig.show()
# %%