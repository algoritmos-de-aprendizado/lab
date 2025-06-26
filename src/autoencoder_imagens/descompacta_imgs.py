import zipfile
import os

# Caminho do arquivo ZIP
zip_path = 'img_align_celeba.zip'

# Pasta onde serão extraídas as imagens
output_folder = 'celebridades_imgs'
os.makedirs(output_folder, exist_ok=True)

# Número de imagens extraídas
max_imagens = 40000

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Lista de todos os arquivos no zip
    todos_os_arquivos = [f for f in zip_ref.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Limita o número de imagens a extrair
    arquivos_a_extrair = todos_os_arquivos[:max_imagens]

    for arquivo in arquivos_a_extrair:
        zip_ref.extract(arquivo, output_folder)

print(f"{len(arquivos_a_extrair)} imagens extraídas: {output_folder}")
