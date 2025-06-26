from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization

def autoencoder_v2(img_shape=(176, 216, 1)):
    # Entrada
    entrada = Input(shape=img_shape)
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(entrada)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    gargalo = MaxPooling2D((2, 2))(x)
    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(gargalo)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    # Saída
    saida = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    modelo = Model(entrada, saida)
    return modelo

# modelo_v2 = autoencoder_v2()
# modelo_v2.summary()