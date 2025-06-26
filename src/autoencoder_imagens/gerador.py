import os
import numpy as np 
from tensorflow.keras.utils import Sequence 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from sklearn.model_selection import train_test_split 

# Gerador de pares de imagens para treinamento de um autoencoder
class ImagePairGenerator(Sequence):
    def __init__(self, arquivos, pasta_ruido, pasta_original, batch_size=32, img_size=(176, 216)):
        self.arquivos = arquivos
        self.pasta_ruido = pasta_ruido
        self.pasta_original = pasta_original
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.arquivos) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.arquivos[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for arquivo in batch_files:
            # Carrega as imagens de ruído e original
            img_ruido = load_img(os.path.join(self.pasta_ruido, arquivo), color_mode='grayscale', target_size=self.img_size)
            img_limp = load_img(os.path.join(self.pasta_original, arquivo), color_mode='grayscale', target_size=self.img_size)
            # Converte as imagens para arrays e normaliza
            arr_ruido = img_to_array(img_ruido) / 255.0
            arr_limp = img_to_array(img_limp) / 255.0
            # Adiciona as imagens ao lote
            X.append(arr_ruido)
            Y.append(arr_limp)
        return np.array(X), np.array(Y)

# Função para dividir o dataset em conjuntos de treino e teste
def dividir_dataset(pasta_ruido):
    arquivos = sorted(os.listdir(pasta_ruido))
    return train_test_split(arquivos, test_size=0.2, random_state=42)
