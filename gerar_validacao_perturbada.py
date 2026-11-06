"""
Geração dos conjuntos perturbados

Aplica as 5 perturbações, em 4 níveis cada, nas imagens de validação ou de teste, gerando 20 versões perturbadas de cada conjunto. Rodar uma vez só.

A validação perturbada é usada para escolher o melhor modelo durante o treino, e o teste perturbado só na avaliação final. Os dois usam semente base e ordem de tipos diferentes, para que nunca recebam o mesmo ruído:
    semente = semente_base + 1000 * indice_do_tipo + round(nivel * 100)
    validação: semente base 777 / teste: semente base 42

Como rodar (dentro da pasta do projeto):
    python gerar_validacao_perturbada.py           (validação)
    python gerar_validacao_perturbada.py --teste   (teste)

"""

import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import perturb_aerial_v3 as pa

PASTA = Path(__file__).resolve().parent
NIVEIS = [0.25, 0.50, 0.75, 1.00]
LIMIAR_BRANCO = 245   # branco (>= 245) ligado à borda da imagem = fundo sem dados, não é perturbado
EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

CONJUNTOS = {
    "valid": {
        "origem": PASTA / "dataset_unet" / "valid" / "imagens",
        "destino": PASTA / "dataset_unet" / "valid_perturbada",
        "semente_base": 777,
        "tipos": ["gaussian", "fog", "sensor", "illumination", "occlusion"],
    },
    "teste": {
        "origem": PASTA / "dataset_unet" / "test" / "imagens",
        "destino": PASTA / "dataset_unet" / "teste_perturbado",
        "semente_base": 42,
        "tipos": ["gaussian", "sensor", "fog", "illumination", "occlusion"],
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teste", action="store_true", help="gera o teste perturbado")
    args = parser.parse_args()

    if args.teste:
        conjunto = CONJUNTOS["teste"]
    else:
        conjunto = CONJUNTOS["valid"]
    origem = conjunto["origem"]
    destino = conjunto["destino"]
    semente_base = conjunto["semente_base"]
    tipos = conjunto["tipos"]

    if destino.exists():
        raise SystemExit(f"A pasta {destino} já existe. Apagar antes de rodar de novo.")
    imagens = sorted(p for p in origem.iterdir() if p.suffix.lower() in EXTENSOES)
    if len(imagens) != 72:
        raise SystemExit(f"Eram esperadas 72 imagens em {origem}, foram encontradas {len(imagens)}.")

    originais = [pa.carregar_imagem(p) for p in imagens]
    # máscara das áreas com imagem (fora do fundo branco)
    validos = [pa.mascara_valida(img, LIMIAR_BRANCO) for img in originais]

    for i, tipo in enumerate(tipos):
        for nivel in NIVEIS:
            # cada tipo e nível tem a sua semente
            semente = semente_base + 1000 * i + int(round(nivel * 100))
            random.seed(semente)
            np.random.seed(semente)

            pasta = destino / f"{tipo}_{nivel:.2f}"
            pasta.mkdir(parents=True)
            for caminho, img, valido in zip(imagens, originais, validos):
                perturbada = pa.PERTURBACOES[tipo](img, nivel)
                perturbada = np.where(valido[..., None], perturbada, img)  # mantém o fundo branco
                cv2.imwrite(str(pasta / (caminho.stem + ".png")), perturbada)
            print(f"{tipo:<13} nível {nivel:.2f}  semente {semente}  -> {pasta.name}")

    print(f"\nConjunto perturbado criado em {destino} (20 conjuntos x 72 imagens)")


if __name__ == "__main__":
    main()