# test_gan.py
import tensorflow as tf 
import numpy as np 
import os
import random
import matplotlib.pyplot as plt 
from tqdm import tqdm 

# --- CONFIGURAÇÕES ---
PASTA_ORIGINAL = './celebridades_imgs_original'
PASTA_RUIDO = './celebridades_imgs_ruido_20'
# Carrega o gerador treinado
MODELO_PATH = './training_checkpoints_gan/generator_final.h5' 
OUTPUT_DIR = './resultados_gan'
LOG_PATH = os.path.join(OUTPUT_DIR, 'avaliacao_gan.txt')
IMG_SIZE = (176, 216)
NUM_AMOSTRAS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CARREGAMENTO E TESTE ---
generator = tf.keras.models.load_model(MODELO_PATH)

arquivos = sorted(os.listdir(PASTA_RUIDO))
amostras_teste = random.sample(arquivos, NUM_AMOSTRAS)

psnr_scores = []
ssim_scores = []

with open(LOG_PATH, 'w') as f_log:
    f_log.write(f"Resultados da Avaliação para o Modelo GAN\n{'-'*50}\nArquivo, PSNR, SSIM\n")
    
    for nome_arquivo in tqdm(amostras_teste, desc="Avaliando modelo GAN"):
        img_r = tf.io.read_file(os.path.join(PASTA_RUIDO, nome_arquivo))
        img_r = tf.image.decode_jpeg(img_r, channels=1)
        img_r = tf.image.resize(img_r, [IMG_SIZE[0], IMG_SIZE[1]])
        arr_r_norm = (img_r / 127.5) - 1 # Normaliza para [-1, 1]

        img_o = tf.io.read_file(os.path.join(PASTA_ORIGINAL, nome_arquivo))
        img_o = tf.image.decode_jpeg(img_o, channels=1)
        arr_o = tf.image.resize(img_o, [IMG_SIZE[0], IMG_SIZE[1]])
        
        # Gera a imagem reconstruída
        reconstruida_norm = generator(tf.expand_dims(arr_r_norm, 0), training=False)[0]
        # Desnormaliza para calcular métricas e salvar
        reconstruida = (reconstruida_norm * 127.5 + 127.5)
        
        # Cálculo das Métricas (em imagens 0-255)
        psnr = tf.image.psnr(tf.cast(arr_o, tf.uint8), tf.cast(reconstruida, tf.uint8), max_val=255).numpy()
        ssim = tf.image.ssim(tf.cast(arr_o, tf.uint8), tf.cast(reconstruida, tf.uint8), max_val=255).numpy()
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        f_log.write(f"{nome_arquivo}, {psnr:.4f}, {ssim:.4f}\n")

        # Salvando a imagem comparativa
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        # Exibe imagens normalizadas para 0-1
        axs[0].imshow(tf.cast(arr_o, tf.uint8).numpy().squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[0].set_title("Original")
        axs[0].axis('off')
        axs[1].imshow(tf.cast(img_r, tf.uint8).numpy().squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[1].set_title("Com Ruído")
        axs[1].axis('off')
        axs[2].imshow(tf.cast(reconstruida, tf.uint8).numpy().squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[2].set_title(f"Reconstruída (GAN)\nPSNR: {psnr:.2f} SSIM: {ssim:.4f}")
        axs[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"comparacao_{nome_arquivo}"))
        plt.close()

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    f_log.write(f"\n{'-'*50}\nMÉDIA GERAL PSNR: {avg_psnr:.4f}\nMÉDIA GERAL SSIM: {avg_ssim:.4f}\n")
    print(f"\nAvaliação concluída. Média PSNR: {avg_psnr:.2f}, Média SSIM: {avg_ssim:.4f}")