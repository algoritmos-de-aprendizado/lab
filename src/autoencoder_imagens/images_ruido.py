import os
import numpy as np 
from keras.preprocessing.image import load_img, img_to_array, array_to_img 

def adicionar_ruido(img_array, ruido_sigma=0.2):
    ruido = np.random.normal(loc=0.0, scale=ruido_sigma, size=img_array.shape)
    img_ruidosa = img_array + ruido
    return np.clip(img_ruidosa, 0., 1.)

# Cria pasta para salvar imagens com ruído
os.makedirs("celebridades_imgs_ruido_20", exist_ok=True)

# Processa uma imagem por vez
pasta_origem = "celebridades_imgs"
pasta_destino = "celebridades_imgs_ruido_20"

for arquivo in os.listdir(pasta_origem):
    if arquivo.lower().endswith('.jpg'):
        caminho = os.path.join(pasta_origem, arquivo)
        img = load_img(caminho, color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_ruido = adicionar_ruido(img_array, ruido_sigma=0.2)
        img_ruido = array_to_img(img_ruido)
        img_ruido.save(os.path.join(pasta_destino, arquivo))

print("Imagens com ruído salvas em 'celebridades_imgs_ruido_200'")
