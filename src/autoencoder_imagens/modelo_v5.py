from keras.models import Model 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization  # Adicionado o concatenate
import tensorflow as tf 

# Função de perda: MSE + (1 - SSIM)
def mse_ssim_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return mse + 0.84 * (1.0 - ssim)

def unet_autoencoder(img_shape=(176, 216, 1)):
    entrada = Input(shape=img_shape)
    # Encoder
    # Bloco 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(entrada)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    # Bloco 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    # Bloco central (gargalo)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    # Decoder
    # Bloco 4
    up4 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1) # Skip connection
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    # Bloco 5
    up5 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1) # Skip connection
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    # Saída
    saida = Conv2D(1, (1, 1), activation='sigmoid')(conv5) # Filtro 1x1 na saída

    modelo = Model(entrada, saida)
    modelo.compile(optimizer='adam', loss=mse_ssim_loss)
    return modelo

# Para usar o modelo
modelo_otimizado = unet_autoencoder()
modelo_otimizado.summary()