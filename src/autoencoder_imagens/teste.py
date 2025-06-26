import os
import random
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from keras.models import load_model 
from keras.preprocessing.image import load_img, img_to_array 
from tqdm import tqdm 

# --- CONFIGURAÇÕES ---
PASTA_ORIGINAL = './celebridades_imgs_original'
PASTA_RUIDO = './celebridades_imgs_ruido_20'
# Lista com o nome de todos os modelos a serem testados
NOMES_MODELOS = [f'autoencoder_ruido_20_v{i}.h5' for i in range(1, 6)]
# Diretório base para salvar todos os resultados
OUTPUT_BASE_DIR = './resultados_testes'
# Quantidade de imagens aleatórias para testar
NUM_AMOSTRAS = 20
IMG_SIZE = (176, 216)

# Garante que o diretório base de saída exista
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# --- INÍCIO DO PROCESSAMENTO ---

# Lista todos os arquivos de imagem disponíveis para o teste
try:
    arquivos_disponiveis = sorted(os.listdir(PASTA_RUIDO))
    if not arquivos_disponiveis:
        raise FileNotFoundError("A pasta de imagens com ruído está vazia ou não foi encontrada.")
    # Seleciona um conjunto fixo de imagens aleatórias para testar em todos os modelos
    if len(arquivos_disponiveis) < NUM_AMOSTRAS:
        print(f"Aviso: Encontrado(s) apenas {len(arquivos_disponiveis)} arquivo(s). Usando todos para o teste.")
        amostras_teste = arquivos_disponiveis
    else:
        amostras_teste = random.sample(arquivos_disponiveis, NUM_AMOSTRAS)
except FileNotFoundError as e:
    print(f"Erro: {e}")
    print("Verifique se as pastas 'celebridades_imgs_original' e 'celebridades_imgs_ruido_20' existem e não estão vazias.")
    exit()

# Loop principal para processar cada modelo
for nome_modelo in NOMES_MODELOS:
    id_modelo = nome_modelo.split('_')[-1].split('.')[0] # Extrai 'v1', 'v2', etc.
    print(f"\n{'='*50}\nProcessando Modelo: {nome_modelo}\n{'='*50}")

    # --- Carregamento e Configuração por Modelo ---
    modelo_path = os.path.join(nome_modelo) # Assumindo que os modelos estão no mesmo diretório
    output_dir_modelo = os.path.join(OUTPUT_BASE_DIR, f'amostras_reconstruidas_{id_modelo}')
    log_path = os.path.join(output_dir_modelo, f'avaliacao_{id_modelo}.txt')
    os.makedirs(output_dir_modelo, exist_ok=True)

    try:
        modelo = load_model(modelo_path, compile=False)
    except (IOError, FileNotFoundError):
        print(f"Erro: Modelo '{nome_modelo}' não encontrado. Pulando para o próximo.")
        continue

    # Listas para armazenar as métricas de cada imagem
    psnr_scores = []
    ssim_scores = []

    # Abre o arquivo de log para escrita
    with open(log_path, 'w') as f_log:
        f_log.write(f"Resultados da Avaliação para o Modelo: {nome_modelo}\n")
        f_log.write("-" * 50 + "\n")
        f_log.write("Arquivo, PSNR, SSIM\n")

        # Loop para processar cada imagem de amostra
        for nome_arquivo in tqdm(amostras_teste, desc=f"Processando Imagens ({id_modelo})"):
            # Carrega a imagem com ruído e a original
            img_r_path = os.path.join(PASTA_RUIDO, nome_arquivo)
            img_o_path = os.path.join(PASTA_ORIGINAL, nome_arquivo)
            
            img_r = load_img(img_r_path, color_mode='grayscale', target_size=IMG_SIZE)
            img_o = load_img(img_o_path, color_mode='grayscale', target_size=IMG_SIZE)

            # Converte imagens para array e normaliza
            arr_r = img_to_array(img_r) / 255.0
            arr_o = img_to_array(img_o) / 255.0

            # Realiza a predição (reconstrução)
            # Adiciona uma dimensão para o batch (lote de 1)
            reconstruida = modelo.predict(np.expand_dims(arr_r, axis=0), verbose=0)[0]

            # --- Cálculo das Métricas ---
            # Garante que as imagens tenham o mesmo tipo de dado para as métricas
            arr_o_tensor = tf.convert_to_tensor(arr_o, dtype=tf.float32)
            reconstruida_tensor = tf.convert_to_tensor(reconstruida, dtype=tf.float32)
            
            psnr = tf.image.psnr(arr_o_tensor, reconstruida_tensor, max_val=1.0).numpy()
            ssim = tf.image.ssim(arr_o_tensor, reconstruida_tensor, max_val=1.0).numpy()
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            
            # Salva os resultados no arquivo de log
            f_log.write(f"{nome_arquivo}, {psnr:.4f}, {ssim:.4f}\n")

            # --- Salvando a imagem comparativa ---
            fig, axs = plt.subplots(1, 3, figsize=(12, 5))
            fig.suptitle(f'Modelo: {id_modelo} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}')
            axs[0].imshow(arr_o.squeeze(), cmap='gray')
            axs[0].set_title("Original")
            axs[0].axis('off')
            axs[1].imshow(arr_r.squeeze(), cmap='gray')
            axs[1].set_title("Com Ruído")
            axs[1].axis('off')
            axs[2].imshow(reconstruida.squeeze(), cmap='gray')
            axs[2].set_title("Reconstruída")
            axs[2].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir_modelo, f"comparacao_{nome_arquivo}"))
            plt.close()

        # --- Cálculo e Log das Médias ---
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)

        print(f"\nResultados para o modelo {id_modelo}:")
        print(f"  PSNR Médio: {avg_psnr:.4f}")
        print(f"  SSIM Médio: {avg_ssim:.4f}")
        print(f"  Resultados salvos em: {output_dir_modelo}")
        
        f_log.write("-" * 50 + "\n")
        f_log.write(f"MÉDIA GERAL PSNR: {avg_psnr:.4f}\n")
        f_log.write(f"MÉDIA GERAL SSIM: {avg_ssim:.4f}\n")

print("\n\nProcesso de avaliação de todos os modelos concluído!")