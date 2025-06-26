# train_gan.py
import tensorflow as tf
import numpy as np 
import os
import time
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from generator import build_generator
from discriminator import build_discriminator

# --- CONFIGURAÇÕES ---
PASTA_ORIGINAL = './celebridades_imgs_original'
PASTA_RUIDO = './celebridades_imgs_ruido_20'
CHECKPOINT_DIR = './training_checkpoints_gan'
LOG_DIR = './logs_gan'
IMG_SIZE = (176, 216)

# Parâmetros de treinamento
EPOCHS = 10 
BATCH_SIZE = 4 # Número de imagens por lote
BUFFER_SIZE = 400 # Tamanho do buffer para embaralhamento
LAMBDA = 100

# --- PREPARAÇÃO DOS DADOS ---
def load_image(noisy_path, original_path):
    noisy_img = tf.io.read_file(noisy_path)
    noisy_img = tf.image.decode_jpeg(noisy_img, channels=1)
    noisy_img = tf.image.resize(noisy_img, [IMG_SIZE[0], IMG_SIZE[1]])
    noisy_img = (noisy_img / 127.5) - 1 # Normalização

    original_img = tf.io.read_file(original_path)
    original_img = tf.image.decode_jpeg(original_img, channels=1)
    original_img = tf.image.resize(original_img, [IMG_SIZE[0], IMG_SIZE[1]])
    original_img = (original_img / 127.5) - 1 # Normalização

    return noisy_img, original_img

# Carrega os caminhos das imagens
noisy_paths = sorted([os.path.join(PASTA_RUIDO, fname) for fname in os.listdir(PASTA_RUIDO)])
original_paths = sorted([os.path.join(PASTA_ORIGINAL, fname) for fname in os.listdir(PASTA_ORIGINAL)])

train_dataset = tf.data.Dataset.from_tensor_slices((noisy_paths, original_paths))
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- CONSTRUÇÃO DOS MODELOS E OTIMIZADORES ---
generator = build_generator()
discriminator = build_discriminator()

# Otimizadores
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(4e-6, beta_1=0.5)

# Função de perda (Loss)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Funções de perda para o Gerador e Discriminador
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, generated_images, real_images):
    # Perda da GAN: o Gerador quer que o Discriminador pense que as imagens são reais
    gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    # Perda L1: a diferença pixel a pixel entre a imagem gerada e a real
    l1_loss = tf.reduce_mean(tf.abs(real_images - generated_images))

    # Combina as duas perdas
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# --- CHECKPOINTS PARA SALVAR O PROGRESSO ---
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- FUNÇÃO DE TREINAMENTO ---
@tf.function
def train_step(noisy_images, real_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noisy_images, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Usa a nova função de perda do gerador
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(fake_output, generated_images, real_images)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Retorna todas as perdas para monitoramento
    return gen_gan_loss, gen_l1_loss, disc_loss

# --- FUNÇÃO PARA GERAR E SALVAR IMAGENS DE TESTE DURANTE O TREINO ---
# Pega algumas imagens fixas para vermos a evolução
test_noisy_batch, test_real_batch = next(iter(train_dataset.take(1)))

def generate_and_save_images(model, epoch, test_input, test_real):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(12, 6))

    display_list = [test_input[0], test_real[0], predictions[0]]
    title = ['Com Ruído', 'Original', 'Reconstruída (GAN)']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Desnormaliza de [-1, 1] para [0, 1] para exibir
        plt.imshow((display_list[i] * 0.5 + 0.5), cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(LOG_DIR, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()

# --- LOOP PRINCIPAL DE TREINAMENTO ---
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        # Listas para guardar as perdas da época
        epoch_gen_gan_loss = []
        epoch_gen_l1_loss = []
        epoch_disc_loss = []

        pbar = tqdm(dataset, desc=f"Epoch {epoch + 1}/{epochs}")
        for noisy_batch, real_batch in pbar:
            g_gan_loss, g_l1_loss, d_loss = train_step(noisy_batch, real_batch)

            # Guarda as perdas
            epoch_gen_gan_loss.append(g_gan_loss)
            epoch_gen_l1_loss.append(g_l1_loss)
            epoch_disc_loss.append(d_loss)

            # Atualiza a barra de progresso com as novas perdas
            pbar.set_postfix({"G_GAN_Loss": f"{g_gan_loss:.4f}", "G_L1_Loss": f"{g_l1_loss:.4f}", "D_Loss": f"{d_loss:.4f}"})

        generate_and_save_images(generator, epoch + 1, test_noisy_batch, test_real_batch)
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Imprime a média das perdas da época
        print(f'\nTempo para a época {epoch + 1} foi de {time.time()-start:.2f} seg')
        print(f'Média da Época -> Loss GAN do Gerador: {np.mean(epoch_gen_gan_loss):.4f}, Loss L1 do Gerador: {np.mean(epoch_gen_l1_loss):.4f}, Loss do Discriminador: {np.mean(epoch_disc_loss):.4f}')

    generator.save(os.path.join(CHECKPOINT_DIR, 'generator_final_pix2pix.h5'))

if __name__ == '__main__':
    train(train_dataset, EPOCHS)