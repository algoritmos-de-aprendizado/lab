from gerador import ImagePairGenerator, dividir_dataset
from autoencoder_imagens.modelo_v5 import unet_autoencoder

batch_size = 32
img_size = (176, 216)

train_files, val_files = dividir_dataset("./celebridades_imgs_ruido_20")


gerador_treino = ImagePairGenerator(train_files,
                                    "./celebridades_imgs_ruido_20",
                                    "./celebridades_imgs_original",
                                    batch_size=batch_size,
                                    img_size=img_size)

gerador_val = ImagePairGenerator(val_files,
                                 "./celebridades_imgs_ruido_20",
                                 "./celebridades_imgs_original",
                                 batch_size=batch_size,
                                 img_size=img_size)

print("Criando autoencoder!\n")

modelo = unet_autoencoder(img_shape=(img_size[0], img_size[1], 1))
modelo.fit(gerador_treino, validation_data=gerador_val, epochs=10)

modelo.save("autoencoder_ruido_20_v5.h5")