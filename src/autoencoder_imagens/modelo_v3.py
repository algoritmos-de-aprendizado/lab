from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape

def autoencoder_denso_v3(img_shape=(176, 216, 1)):
    input_dim = img_shape[0] * img_shape[1]
    
    entrada = Input(shape=img_shape)
    # Encoder
    x = Flatten()(entrada)
    x = Dense(1200, activation='relu')(x)
    x = Dense(800, activation='relu')(x)
    x = Dense(400, activation='relu')(x)
    gargalo = Dense(100, activation='relu')(x)
    # Decoder
    x = Dense(400, activation='relu')(gargalo)
    x = Dense(800, activation='relu')(x)
    x = Dense(1200, activation='relu')(x)
    x = Dense(input_dim, activation='sigmoid')(x)
    saida = Reshape(img_shape)(x)

    modelo = Model(entrada, saida)
    return modelo

# modelo_v3 = autoencoder_denso_v3()
# modelo_v3.summary()